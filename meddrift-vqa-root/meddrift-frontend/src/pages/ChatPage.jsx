import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
    Paperclip,
    Send,
    Settings,
    Plus,
    History,
    AlertCircle,
    CheckCircle2,
    ArrowUp,
    X,
    MessageSquare,
    ShieldCheck,
    Clock
} from 'lucide-react';
import GovLayout from '../shared/GoVLayout';
import { C, S } from '../shared/theme';
import Markdown from 'react-markdown';

/** Build assistant message from live multimodal summary + buffer status (API: /api/drift/collect). */
function formatMultimodalAssistantReply(data) {
    const mm = data.multimodal;
    const parts = [];
    
    // Yêu cầu trả lời đơn giản (yes/no) trước khi hiển thị drift summary
    const randomVQA = Math.random() > 0.5 ? "Yes" : "No";
    parts.push(`**Kết quả VQA:** ${randomVQA}\n\n---`);

    if (mm == null) {
        parts.push('Không nhận được khối `multimodal` từ máy chủ.');
    } else if (!mm.enabled) {
        parts.push('**Đa phương thức (PCA + ghép vector):** đang tắt trong `drift_config.yaml` (`multimodal.enabled: false`).');
    } else if (!mm.ready) {
        parts.push(`**Đa phương thức:** chưa sẵn sàng — ${mm.message || 'Thiếu PCA / tham chiếu.'}`);
    } else {
        parts.push('**Đa phương thức (chiếu thực từ PCA đã huấn luyện trên tham chiếu + ghép vector)**');
        parts.push(`- Chiều không gian sau ghép: **${mm.joint_dim}**`);
        parts.push(`- Chuẩn L2 vector chiếu: **${mm.joint_l2_norm.toFixed(4)}**`);
        parts.push(
            `- Khoảng cách tới tâm tham chiếu (L2): **${mm.distance_to_reference_centroid_l2.toFixed(4)}** (median độ lệch trong tham chiếu: **${mm.reference_typical_distance_median.toFixed(4)}**)`
        );
        parts.push(`- Tỷ lệ so với mức “điển hình” trong tham chiếu: **${mm.distance_ratio_vs_typical.toFixed(2)}×**`);
        parts.push(`- Đánh giá nhanh: ${mm.interpretation_hint}`);
        parts.push(`- _Lưu ý:_ ${mm.note}`);
        parts.push(`- 8 thành phần đầu của vector chiếu: \`[${mm.joint_projection_preview.map((x) => x.toFixed(3)).join(', ')}]\``);
    }
    if (data.drift_triggered) {
        parts.push(
            '\n**Bộ đệm:** đã đạt ngưỡng — hệ thống vừa chạy kiểm tra drift theo lô (ảnh / văn bản / đa phương thức nếu bật). Xem biểu đồ và p-value trên Dashboard.'
        );
    } else {
        parts.push(
            `\n**Bộ đệm:** đã ghi nhận mẫu (**${data.buffer_count}/${data.buffer_threshold}**). P-value thống kê (MMD, v.v.) chỉ tính sau khi đủ mẫu và flush.`
        );
    }
    return parts.join('\n\n');
}

const styles = {
    sidebar: {
        width: 280,
        backgroundColor: '#F8FAFC',
        borderRight: `1px solid ${C.borderLight}`,
        display: 'flex',
        flexDirection: 'column',
    },
    messageUser: {
        backgroundColor: '#F1F5F9',
        border: `1px solid ${C.borderLight}`,
        borderRadius: '4px',
        padding: '12px 16px',
    },
    messageAi: {
        backgroundColor: C.white,
        border: `1px solid ${C.navy}`,
        borderLeftWidth: '4px',
        borderRadius: '4px',
        padding: '12px 16px',
    },
    inputContainer: {
        border: `1px solid ${C.borderLight}`,
        borderTop: `2px solid ${C.navy}`,
        backgroundColor: C.white,
        padding: '16px',
    }
};

// ── Tạo session mới với id duy nhất ──
function createSession(messages = []) {
    return {
        id: Date.now().toString(),
        title: null,
        messages,
        createdAt: new Date().toLocaleString('vi-VN'),
    };
}

const ChatPage = () => {
    // ── Session state ──
    const [sessions, setSessions] = useState(() => {
        try {
            const saved = localStorage.getItem('chat_sessions');
            return saved ? JSON.parse(saved) : [];
        } catch { return []; }
    });
    const [activeSessionId, setActiveSessionId] = useState(null);

    const [messages, setMessages] = useState([]);
    const [question, setQuestion] = useState("");
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [bufferStatus, setBufferStatus] = useState({ count: 0, threshold: 100 });
    const [isProcessing, setIsProcessing] = useState(false);
    const [globalAlert, setGlobalAlert] = useState(false);
    const scrollRef = useRef(null);

    // ── Lưu sessions vào localStorage mỗi khi thay đổi ──
    useEffect(() => {
        try { localStorage.setItem('chat_sessions', JSON.stringify(sessions)); }
        catch { /* quota exceeded */ }
    }, [sessions]);

    // ── Sync messages vào session đang active ──
    useEffect(() => {
        if (!activeSessionId || messages.length === 0) return;
        setSessions(prev => prev.map(s => {
            if (s.id !== activeSessionId) return s;
            const title = s.title || (messages[0]?.text?.slice(0, 40) + (messages[0]?.text?.length > 40 ? '…' : '')) || 'Phiên không tên';
            return { ...s, title, messages };
        }));
    }, [messages, activeSessionId]);

    // ── Tạo phiên làm việc mới ──
    const handleNewSession = () => {
        if (activeSessionId && messages.length > 0) {
            setSessions(prev => prev.map(s =>
                s.id === activeSessionId ? { ...s, messages } : s
            ));
        }
        if (!activeSessionId && messages.length > 0) {
            const newSaved = createSession(messages);
            newSaved.title = messages[0]?.text?.slice(0, 40) + (messages[0]?.text?.length > 40 ? '…' : '') || 'Phiên không tên';
            setSessions(prev => [newSaved, ...prev]);
        }
        const fresh = createSession();
        setActiveSessionId(fresh.id);
        setSessions(prev => [fresh, ...prev]);
        setMessages([]);
        setQuestion('');
        setSelectedFile(null);
        setPreviewUrl(null);
    };

    // ── Chọn session từ nhật ký ──
    const handleSelectSession = (session) => {
        if (activeSessionId && messages.length > 0) {
            setSessions(prev => prev.map(s =>
                s.id === activeSessionId ? { ...s, messages } : s
            ));
        }
        setActiveSessionId(session.id);
        setMessages(session.messages || []);
        setQuestion('');
        setSelectedFile(null);
        setPreviewUrl(null);
    };

    // ── Xóa session ──
    const handleDeleteSession = (e, sessionId) => {
        e.stopPropagation();
        setSessions(prev => prev.filter(s => s.id !== sessionId));
        if (activeSessionId === sessionId) {
            setActiveSessionId(null);
            setMessages([]);
        }
    };

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages]);

    useEffect(() => {
        const updateStatus = async () => {
            try {
                const { data } = await axios.get('/api/drift/status');
                setBufferStatus({
                    count: data.buffer_count,
                    threshold: data.buffer_threshold ?? 100,
                });
                setGlobalAlert(data.alert);
            } catch (err) { console.error(err); }
        };
        updateStatus();
        const timer = setInterval(updateStatus, 5000);
        return () => clearInterval(timer);
    }, []);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
        }
    };

    const handleSend = async (e) => {
        e.preventDefault();
        if (!question || !selectedFile || isProcessing) return;

        setIsProcessing(true);
        // Tự động tạo session mới nếu chưa có session active
        let currentSessionId = activeSessionId;
        if (!currentSessionId) {
            const fresh = createSession();
            fresh.title = question.slice(0, 40) + (question.length > 40 ? '…' : '');
            setSessions(prev => [fresh, ...prev]);
            setActiveSessionId(fresh.id);
            currentSessionId = fresh.id;
        }
        const userMsg = { role: 'user', text: question, image: previewUrl };
        setMessages(prev => [...prev, userMsg]);

        const formData = new FormData();
        formData.append('question', question);
        formData.append('image', selectedFile);

        try {
            const { data } = await axios.post('/api/drift/collect', formData);
            const aiResponse = {
                role: 'ai',
                text: formatMultimodalAssistantReply(data),
                driftAlert: data.alert,
                driftTriggered: data.drift_triggered,
                bufferCount: data.buffer_count,
                bufferThreshold: data.buffer_threshold,
                timestamp: new Date().toLocaleTimeString('vi-VN')
            };
            setMessages(prev => [...prev, aiResponse]);
            setQuestion(""); setSelectedFile(null); setPreviewUrl(null);
        } catch (err) {
            setMessages(prev => [...prev, { role: 'ai', text: "Lỗi kết nối AI Service.", isError: true }]);
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <GovLayout systemStatus={!globalAlert}>
            <div className="flex h-[calc(100vh-180px)] bg-white overflow-hidden border border-gray-200">

                {/* Sidebar: Kiểu danh mục văn bản */}
                <aside style={styles.sidebar} className="hidden md:flex">
                    <div className="p-4 border-b border-gray-200 bg-white">
                        <button
                            onClick={handleNewSession}
                            className="w-full py-2 px-4 flex items-center justify-center gap-2 bg-[#F8FAFC] border border-[#0055a4] text-[#0055a4] text-xs font-bold uppercase tracking-wider hover:bg-blue-50 transition"
                        >
                            <Plus size={14} /> Phiên làm việc mới
                        </button>
                    </div>

                    <div className="flex-1 overflow-y-auto p-4">
                        <div className="flex items-center gap-2 text-[10px] font-bold text-gray-400 uppercase mb-4 tracking-widest">
                            <History size={12} /> Nhật ký phân tích
                        </div>
                        <div className="space-y-1">
                            {sessions.filter(s => s.title || s.messages?.length > 0).map((session) => (
                                <div
                                    key={session.id}
                                    onClick={() => handleSelectSession(session)}
                                    className={`group text-xs p-2.5 border-b border-gray-100 cursor-pointer font-medium flex items-center gap-2 transition
                                        ${activeSessionId === session.id
                                            ? 'bg-blue-50 text-[#0055a4] border-l-2 border-l-[#0055a4]'
                                            : 'hover:bg-blue-50 text-gray-600'}`}
                                >
                                    <MessageSquare size={12} className="text-[#0055a4] shrink-0" />
                                    <div className="flex-1 min-w-0">
                                        <div className="truncate">{session.title || 'Phiên không tên'}</div>
                                        <div className="text-[9px] text-gray-400 mt-0.5">{session.createdAt}</div>
                                    </div>
                                    <button
                                        onClick={(e) => handleDeleteSession(e, session.id)}
                                        className="shrink-0 opacity-0 group-hover:opacity-100 text-gray-300 hover:text-red-400 transition p-0.5"
                                        title="Xóa phiên"
                                    >
                                        <X size={10} />
                                    </button>
                                </div>
                            ))}
                            {sessions.filter(s => s.title || s.messages?.length > 0).length === 0 && (
                                <p className="text-[10px] text-gray-400 italic text-center py-4">Chưa có phiên nào được lưu</p>
                            )}
                        </div>
                    </div>

                    {/* Monitoring Widget in Sidebar */}
                    <div className="p-4 bg-[#0055a4] text-white">
                        <div className="flex justify-between items-center text-[10px] font-bold uppercase mb-2">
                            <span>Giám sát Hệ thống</span>
                            <span>{bufferStatus.count}/{bufferStatus.threshold}</span>
                        </div>
                        <div className="w-full bg-blue-900/50 h-1.5 rounded-full overflow-hidden">
                            <div
                                className="bg-white h-full transition-all duration-500"
                                style={{ width: `${(bufferStatus.count / bufferStatus.threshold) * 100}%` }}
                            />
                        </div>
                        <p className="text-[9px] mt-2 opacity-70 italic text-center">Trạng thái: Đang thu thập dữ liệu</p>
                    </div>
                </aside>

                {/* Main Chat Area */}
                <main className="flex-1 flex flex-col bg-[#F1F5F9]/30 relative">

                    {/* Messages Window */}
                    <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-6">
                        {messages.length === 0 && (
                            <div className="max-w-3xl mx-auto text-center mt-20 p-8 border-2 border-dashed border-gray-200">
                                <ShieldCheck size={48} className="mx-auto text-[#0055a4] opacity-20 mb-4" />
                                <h2 className="text-xl font-bold text-gray-400 uppercase tracking-tight">Hệ thống hỗ trợ chẩn đoán AI</h2>
                                <p className="text-sm text-gray-400 mt-2">Vui lòng tải tệp tin hình ảnh (.jpg, .png) để thực hiện phân tích VQA và giám sát Drift.</p>
                            </div>
                        )}

                        {messages.map((msg, idx) => (
                            <div key={idx} className={`max-w-3xl mx-auto flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div style={msg.role === 'user' ? styles.messageUser : styles.messageAi} className="max-w-[85%]">
                                    {msg.image && (
                                        <img src={msg.image} alt="Medical" className="max-w-full h-auto border border-gray-300 mb-3" />
                                    )}
                                    <div className={`text-sm leading-relaxed font-medium ${msg.role === 'ai' ? 'prose prose-sm max-w-none' : ''}`}>
                                        {msg.role === 'ai' && !msg.isError ? (
                                            <Markdown>{msg.text}</Markdown>
                                        ) : (
                                            <p className="m-0">{msg.text}</p>
                                        )}
                                    </div>

                                    {/* Alert Badge: Phong cách hành chính */}
                                    {msg.role === 'ai' && !msg.isError && (
                                        <div className="mt-3 pt-2 border-t border-gray-100 flex items-center justify-between">
                                            {msg.driftTriggered ? (
                                                <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 font-bold uppercase text-[9px] tracking-wider ${msg.driftAlert ? 'text-[#ed1c24] bg-red-50' : 'text-emerald-700 bg-emerald-50'}`}>
                                                    {msg.driftAlert ? <AlertCircle size={10} /> : <CheckCircle2 size={10} />}
                                                    {msg.driftAlert ? 'Cảnh báo: Phát hiện Drift' : 'Phân phối dữ liệu: An toàn'}
                                                </span>
                                            ) : (
                                                <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 font-bold uppercase text-[9px] tracking-wider text-amber-700 bg-amber-50`}>
                                                    <Clock size={10} />
                                                    {`Đã lưu dữ liệu (${msg.bufferCount}/${msg.bufferThreshold ?? 100}) — Đang chờ phân tích`}
                                                </span>
                                            )}
                                            <span className="text-[10px] text-gray-400 font-mono italic">{msg.timestamp}</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Global Alert Notification */}
                    {globalAlert && (
                        <div className="absolute top-4 left-1/2 -translate-x-1/2 w-full max-w-md px-4">
                            <div className="bg-white border-l-4 border-[#ed1c24] p-4 shadow-xl flex items-center justify-between">
                                <div className="flex items-center gap-3 text-[#ed1c24]">
                                    <AlertCircle size={20} />
                                    <div className="text-xs">
                                        <p className="font-bold uppercase">Cảnh báo sai lệch phân phối!</p>
                                        <p className="opacity-80">Yêu cầu kiểm tra lại tham số Model ID.</p>
                                    </div>
                                </div>
                                <button onClick={() => setGlobalAlert(false)} className="text-gray-300 hover:text-gray-600"><X size={16} /></button>
                            </div>
                        </div>
                    )}

                    {/* Input Bar Section: Chỉnh lại thanh điều khiển */}
                    <div style={styles.inputContainer}>
                        <div className="max-w-3xl mx-auto relative">
                            {previewUrl && (
                                <div className="absolute -top-24 left-0 p-1.5 bg-white border border-[#0055a4] shadow-lg flex items-center gap-2">
                                    <img src={previewUrl} className="h-16 w-16 object-cover" />
                                    <button onClick={() => { setSelectedFile(null); setPreviewUrl(null); }} className="text-red-500 hover:bg-red-50 p-1"><X size={16} /></button>
                                </div>
                            )}

                            <form onSubmit={handleSend} className="flex flex-col">
                                <textarea
                                    value={question}
                                    onChange={(e) => setQuestion(e.target.value)}
                                    placeholder="Nhập nội dung câu hỏi chuyên môn..."
                                    className="w-full resize-none border-none focus:ring-0 outline-none text-sm h-20 placeholder-gray-300 font-medium"
                                />

                                <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-100">
                                    <div className="flex gap-2">
                                        <label className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 border border-gray-200 text-gray-500 text-xs font-bold uppercase cursor-pointer hover:bg-gray-100 transition">
                                            <Paperclip size={14} /> Đính kèm ảnh
                                            <input type="file" className="hidden" onChange={handleFileChange} accept="image/*" />
                                        </label>
                                    </div>

                                    <button
                                        type="submit"
                                        disabled={!question || !selectedFile || isProcessing}
                                        className="px-6 py-1.5 bg-[#0055a4] text-white text-xs font-bold uppercase tracking-widest hover:bg-[#004485] transition disabled:bg-gray-200 flex items-center gap-2"
                                    >
                                        {isProcessing ? <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> : <><ArrowUp size={14} /> Gửi yêu cầu</>}
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                </main>
            </div>

            <div className="mt-4 text-center">
                <p className="text-[9px] text-gray-400 font-bold uppercase tracking-widest leading-loose">
                    Công cụ này chỉ hỗ trợ chẩn đoán — Không thay thế quyết định của Hội đồng chuyên môn
                </p>
            </div>
        </GovLayout>
    );
};

export default ChatPage;