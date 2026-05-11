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
    X
} from 'lucide-react';

const ChatPage = () => {
    // State quản lý tin nhắn và input
    const [messages, setMessages] = useState([]);
    const [question, setQuestion] = useState("");
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);

    // State quản lý trạng thái hệ thống từ FastAPI
    const [bufferStatus, setBufferStatus] = useState({ count: 0, threshold: 100 });
    const [isProcessing, setIsProcessing] = useState(false);
    const [globalAlert, setGlobalAlert] = useState(false);

    const scrollRef = useRef(null);

    // Tự động cuộn xuống khi có tin nhắn mới
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    // Polling trạng thái buffer mỗi 5 giây
    useEffect(() => {
        const updateStatus = async () => {
            try {
                const { data } = await axios.get('/api/drift/status');
                setBufferStatus({ count: data.buffer_count, threshold: data.buffer_threshold });
                setGlobalAlert(data.alert);
            } catch (err) {
                console.error("Lỗi cập nhật trạng thái:", err);
            }
        };
        const timer = setInterval(updateStatus, 5000);
        updateStatus();
        return () => clearInterval(timer);
    }, []);

    // Xử lý chọn file ảnh
    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
        }
    };

    // Gửi yêu cầu VQA và Monitor
    const handleSend = async (e) => {
        e.preventDefault();
        if (!question || !selectedFile || isProcessing) return;

        setIsProcessing(true);
        const userMsg = { role: 'user', text: question, image: previewUrl };
        setMessages(prev => [...prev, userMsg]);

        const formData = new FormData();
        formData.append('question', question);
        formData.append('image', selectedFile);

        try {
            // Gọi API collect để lưu embedding vào buffer
            const { data } = await axios.post('/api/drift/collect', formData);

            // Giả lập câu trả lời từ VQA service (trong thực tế sẽ lấy từ model VQA của bạn)
            const aiResponse = {
                role: 'ai',
                text: "Hệ thống đã nhận được dữ liệu. Dựa trên phân tích sơ bộ, hình ảnh có dấu hiệu tương ứng với các triệu chứng lâm sàng bạn đã nêu.",
                driftAlert: data.alert, // Lấy alert flag từ backend
                timestamp: new Date().toLocaleTimeString()
            };

            setMessages(prev => [...prev, aiResponse]);
            setQuestion("");
            setSelectedFile(null);
            setPreviewUrl(null);
        } catch (err) {
            setMessages(prev => [...prev, { role: 'ai', text: "Lỗi kết nối AI Service.", isError: true }]);
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div className="flex h-screen bg-[#f9f8f6] text-gray-800">

            {/* Sidebar: Lịch sử và Trạng thái */}
            <aside className="w-64 bg-[#f0eee5] border-r border-gray-200 hidden md:flex flex-col">
                <div className="p-4 border-b border-gray-300">
                    <button className="w-full py-2 px-4 flex items-center justify-center gap-2 border border-gray-400 rounded-lg text-sm font-medium hover:bg-gray-200 transition">
                        <Plus size={16} /> Cuộc hội thoại mới
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-4">
                    <div className="flex items-center gap-2 text-xs font-semibold text-gray-500 uppercase mb-4">
                        <History size={14} /> Gần đây
                    </div>
                    <div className="space-y-1">
                        {['Phân tích X-ray phổi', 'Sàng lọc model drift'].map((item, idx) => (
                            <div key={idx} className="text-sm p-2 hover:bg-gray-200 rounded cursor-pointer truncate">
                                {item}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="p-4 border-t border-gray-300 space-y-3">
                    <div className="flex justify-between items-center text-[11px] text-gray-500 uppercase">
                        <span>Buffer Monitoring</span>
                        <span>{bufferStatus.count}/{bufferStatus.threshold}</span>
                    </div>
                    <div className="w-full bg-gray-300 h-1 rounded-full overflow-hidden">
                        <div
                            className="bg-blue-600 h-full transition-all duration-500"
                            style={{ width: `${(bufferStatus.count / bufferStatus.threshold) * 100}%` }}
                        />
                    </div>
                </div>
            </aside>

            {/* Main Chat Area */}
            <main className="flex-1 flex flex-col relative bg-white">

                {/* Top Header */}
                <header className="px-6 py-4 flex justify-between items-center border-b border-gray-100 shadow-sm z-10 bg-white">
                    <div className="font-semibold text-lg text-orange-900 tracking-tight">MedDrift Sentinel</div>
                    <div className="flex items-center gap-4 text-gray-500">
                        <Settings size={20} className="hover:text-gray-800 cursor-pointer" />
                        <div className="w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center text-orange-800 text-xs font-bold">
                            BS
                        </div>
                    </div>
                </header>

                {/* Messages Window */}
                <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-8">
                    {messages.length === 0 && (
                        <div className="max-w-2xl mx-auto text-center mt-32 space-y-4">
                            <h2 className="text-3xl font-light text-gray-400">Tôi có thể giúp gì cho bác sĩ hôm nay?</h2>
                            <p className="text-sm text-gray-400">Tải ảnh X-ray và đặt câu hỏi để bắt đầu phân tích VQA.</p>
                        </div>
                    )}

                    {messages.map((msg, idx) => (
                        <div key={idx} className={`max-w-2xl mx-auto flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`group relative p-4 rounded-2xl shadow-sm border border-gray-100 ${msg.role === 'user' ? 'bg-gray-100' : 'bg-white'}`}>
                                {msg.image && (
                                    <img src={msg.image} alt="Medical" className="max-w-sm rounded-lg mb-3 border border-gray-200" />
                                )}
                                <p className="text-[15px] leading-relaxed">{msg.text}</p>

                                {/* Drift Alert Badge */}
                                {msg.role === 'ai' && !msg.isError && (
                                    <div className="mt-3 flex items-center gap-2">
                                        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-tighter ${msg.driftAlert ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'
                                            }`}>
                                            {msg.driftAlert ? <AlertCircle size={10} /> : <CheckCircle2 size={10} />}
                                            {msg.driftAlert ? 'Critical Drift' : 'Safe Distribution'}
                                        </span>
                                        <span className="text-[10px] text-gray-400">{msg.timestamp}</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Global Drift Warning */}
                {globalAlert && (
                    <div className="absolute top-20 left-1/2 -translate-x-1/2 w-full max-w-lg px-4 animate-bounce">
                        <div className="bg-red-50 border border-red-200 p-3 rounded-xl shadow-lg flex items-center justify-between">
                            <div className="flex items-center gap-3 text-red-700">
                                <AlertCircle size={20} />
                                <div className="text-xs">
                                    <p className="font-bold">Cảnh báo Drift!</p>
                                    <p>Phát hiện sự thay đổi lớn trong phân phối dữ liệu đầu vào.</p>
                                </div>
                            </div>
                            <button onClick={() => setGlobalAlert(false)} className="text-red-400 hover:text-red-600">
                                <X size={16} />
                            </button>
                        </div>
                    </div>
                )}

                {/* Input Bar Section */}
                <div className="px-6 pb-6 bg-white">
                    <div className="max-w-2xl mx-auto relative">

                        {/* Ảnh đang chờ gửi */}
                        {previewUrl && (
                            <div className="absolute -top-24 left-0 p-2 bg-white border border-gray-200 rounded-xl shadow-md flex items-center gap-2">
                                <img src={previewUrl} className="h-16 w-16 object-cover rounded-md" />
                                <button onClick={() => { setSelectedFile(null); setPreviewUrl(null); }} className="text-gray-400 hover:text-red-500">
                                    <X size={16} />
                                </button>
                            </div>
                        )}

                        <div className="border border-gray-300 rounded-3xl p-4 shadow-sm focus-within:border-orange-300 focus-within:ring-4 focus-within:ring-orange-50/50 transition-all">
                            <form onSubmit={handleSend}>
                                <textarea
                                    value={question}
                                    onChange={(e) => setQuestion(e.target.value)}
                                    placeholder="Hỏi MedDrift về hình ảnh y khoa..."
                                    className="w-full resize-none border-none focus:ring-0 outline-none text-gray-700 h-24 placeholder-gray-400"
                                />

                                <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-50">
                                    <div className="flex gap-1">
                                        <label className="p-2 text-gray-400 hover:text-orange-700 hover:bg-orange-50 rounded-full transition cursor-pointer">
                                            <Paperclip size={20} />
                                            <input type="file" className="hidden" onChange={handleFileChange} accept="image/*" />
                                        </label>
                                    </div>

                                    <button
                                        type="submit"
                                        disabled={!question || !selectedFile || isProcessing}
                                        className="p-2 bg-orange-800 text-white rounded-full hover:bg-orange-900 transition disabled:bg-gray-200 disabled:text-gray-400"
                                    >
                                        {isProcessing ? <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" /> : <ArrowUp size={20} />}
                                    </button>
                                </div>
                            </form>
                        </div>
                        <p className="text-[10px] text-gray-400 text-center mt-3 uppercase tracking-tighter">
                            Công cụ hỗ trợ chẩn đoán AI Sentinel v1.0 — HCMUS IT Project
                        </p>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default ChatPage;