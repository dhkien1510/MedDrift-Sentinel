import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Settings, Save, RefreshCw, Layers, Cpu, Activity, ShieldCheck, AlertCircle } from 'lucide-react';
import GovLayout from '../shared/GoVLayout';
import { C, S } from '../shared/theme';

const styles = {
    formGroup: {
        marginBottom: '20px',
    },
    label: {
        display: 'block',
        fontSize: '11px',
        fontWeight: 700,
        color: C.textMuted,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        marginBottom: '6px',
        fontFamily: 'Arial, sans-serif',
    },
    select: {
        width: '100%',
        padding: '10px 12px',
        fontSize: '13px',
        border: `1px solid ${C.borderLight}`,
        borderRadius: '4px',
        backgroundColor: '#F8FAFC',
        outline: 'none',
        fontFamily: 'Arial, sans-serif',
        color: C.navyDark,
    },
    saveButton: {
        backgroundColor: C.navy,
        color: C.white,
        padding: '10px 24px',
        fontSize: '12px',
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        transition: 'background 0.2s',
    }
};

const ENCODER_LABELS = {
    "dmis-lab/biobert-v1.1":                                          "BioBERT v1.1",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract":           "BiomedBERT (Abstract)",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext":  "BiomedBERT (Fulltext)",
    "NeuML/pubmedbert-base-embeddings":                               "PubMedBERT Embeddings",
    "NeuML/pubmedbert-base-embeddings-matryoshka":                    "PubMedBERT Matryoshka",
    "NeuML/pubmedbert-base-embeddings-8M":                            "PubMedBERT 8M (Static)",
    "pritamdeka/S-PubMedBert-MS-MARCO":                               "S-PubMedBERT MS-MARCO",
    "openai/clip-vit-base-patch32":                                   "CLIP ViT-B/32",
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":      "BiomedCLIP ViT-B/16",
    "microsoft/rad-dino":                                             "RAD-DINO",
    "microsoft/rad-dino-maira-2":                                     "RAD-DINO MAIRA-2",
    "facebook/dinov2-base":                                           "DINOv2 Base",
    "google/vit-base-patch16-224":                                    "ViT-B/16 (ImageNet)",
};

const formatEncoderLabel = (enc) => ENCODER_LABELS[enc] ?? enc;

const ConfigPage = () => {
    const [loading, setLoading] = useState(true);
    const [options, setOptions] = useState({ algorithms: [], encoders: [] });
    const [config, setConfig] = useState({
        buffer_threshold: 100,

        image_encoder: '',
        image_algorithm: '',
        image_p_threshold: 0.05,
        image_encoders: [],


        text_encoder: '',
        text_algorithm: '',
        text_p_threshold: 0.05,
        text_encoders: [],
    });

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [algoRes, configRes] = await Promise.all([
                    axios.get('/api/drift/algorithms'),
                    axios.get('/api/drift/config')
                ]);
                setOptions({
                    image_algorithms: algoRes.data.image_algorithms || [],  // ✅
                    text_algorithms: algoRes.data.text_algorithms || [],     // ✅
                    image_encoders: algoRes.data.image_encoders || [],  // thêm
                    text_encoders: algoRes.data.text_encoders || [],  // thêm
                });
                setConfig(configRes.data);
            } catch (err) {
                console.error("Lỗi nạp cấu hình:", err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    const handleSave = async () => {
        try {
            setLoading(true);
            await axios.post('/api/drift/config/apply', config);
            alert("Hệ thống: Cấu hình tham số đã được áp dụng thành công.");
        } catch (err) {
            alert("Lỗi: Không thể kết nối với AI Service để cập nhật.");
        } finally {
            setLoading(false);
        }
    };

    if (loading && !config.image_encoder) {
        return (
            <GovLayout>
                <div className="flex flex-col items-center justify-center h-64">
                    <RefreshCw className="animate-spin text-[#0055a4] mb-4" size={32} />
                    <p className="text-sm font-bold text-gray-400 uppercase tracking-widest">Đang kết nối Registry...</p>
                </div>
            </GovLayout>
        );
    }

    return (
        <GovLayout>
            <div className="max-w-5xl mx-auto space-y-6">
                {/* Header: Phong cách hành chính chuyên nghiệp */}
                <header className="flex justify-between items-end border-b-2 border-gray-100 pb-4">
                    <div>
                        <div className="flex items-center gap-2 text-[#0055a4] mb-1">
                            <Settings size={20} />
                            <h1 className="text-xl font-black uppercase tracking-tight">Thiết lập tham số hệ thống</h1>
                        </div>

                    </div>
                    <button
                        onClick={handleSave}
                        style={styles.saveButton}
                        className="hover:bg-[#004485] active:scale-95 transition-all shadow-sm"
                    >
                        <Save size={16} /> Lưu & Áp dụng thay đổi
                    </button>
                </header>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Panel: Image Pipeline */}
                    <div style={S.panel}>
                        <div style={S.panelHeader}>
                            <div style={S.panelTitle}>Trích xuất đặc trưng hình ảnh (Image)</div>
                        </div>
                        <div className="p-6">
                            <div style={styles.formGroup}>
                                <label style={styles.label}>Image Encoder</label>
                                <select
                                    style={styles.select}
                                    value={config.image_encoder}
                                    onChange={(e) => setConfig({ ...config, image_encoder: e.target.value })}
                                >
                                    {options.image_encoders.map(enc => (
                                        <option key={enc} value={enc}>
                                            {formatEncoderLabel(enc)}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Panel: Text Pipeline */}
                    <div style={S.panel}>
                        <div style={S.panelHeader}>
                            <div style={S.panelTitle}>Trích xuất đặc trưng văn bản (Text)</div>
                        </div>
                        <div className="p-6">
                            <div style={styles.formGroup}>
                                <label style={styles.label}>Text Encoder (Domain Specific BERT)</label>
                                <select
                                    style={styles.select}
                                    value={config.text_encoder}
                                    onChange={(e) => setConfig({ ...config, text_encoder: e.target.value })}
                                >
                                    {options.text_encoders.map(enc => (
                                        <option key={enc} value={enc}>
                                            {formatEncoderLabel(enc)}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Panel: Drift Detection */}
                    <div style={{ ...S.panel, gridColumn: 'span 2' }}>
                        <div style={S.panelHeader}>
                            <div style={S.panelTitle}>Cấu hình giám sát sai lệch</div>
                        </div>
                        <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">

                            {/* Image Algorithm */}
                            <div style={styles.formGroup}>
                                <label style={styles.label}>Thuật toán - Image pipeline</label>
                                <select
                                    style={styles.select}
                                    value={config.image_algorithm}
                                    onChange={(e) => setConfig({ ...config, image_algorithm: e.target.value })}
                                >
                                    {options.image_algorithms?.map(algo => (
                                        <option key={algo} value={algo}>{algo.toUpperCase()}</option>
                                    ))}
                                </select>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', marginBottom: '6px' }}>
                                    <label style={styles.label}>P-Value threshold</label>
                                    <span className="text-lg font-black text-[#0055a4]">{config.image_p_threshold}</span>
                                </div>
                                <input
                                    type="range" min="0.01" max="0.20" step="0.01"
                                    className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#0055a4]"
                                    value={config.image_p_threshold}
                                    onChange={(e) => setConfig({ ...config, image_p_threshold: parseFloat(e.target.value) })}
                                />
                                <div className="flex justify-between text-[9px] font-bold text-gray-400 mt-2 uppercase tracking-tighter">
                                    <span className="text-[#ed1c24]">Khắt khe (0.01)</span>
                                    <span>Trung bình</span>
                                    <span>Lỏng lẻo (0.20)</span>
                                </div>

                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', marginBottom: '6px' }}>
                                    <label style={styles.label}>Kích thước bộ đệm (Buffer Threshold)</label>
                                    <span className="text-lg font-black text-[#0055a4]">{config.buffer_threshold}</span>
                                </div>
                                <input
                                    type="range" min="10" max="500" step="10"
                                    className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#0055a4]"
                                    value={config.buffer_threshold}
                                    onChange={(e) => setConfig({ ...config, buffer_threshold: parseInt(e.target.value) })}
                                />
                                <div className="flex justify-between text-[9px] font-bold text-gray-400 mt-2 uppercase tracking-tighter">
                                    <span className="text-[#ed1c24]">Nhạy (10)</span>
                                    <span>Trung bình</span>
                                    <span>Chậm (500)</span>
                                </div>
                            </div>

                            {/* Text Algorithm */}
                            <div style={styles.formGroup}>
                                <label style={styles.label}>Thuật toán - Text pipeline</label>
                                <select
                                    style={styles.select}
                                    value={config.text_algorithm}
                                    onChange={(e) => setConfig({ ...config, text_algorithm: e.target.value })}
                                >
                                    {options.text_algorithms?.map(algo => (
                                        <option key={algo} value={algo}>{algo.toUpperCase()}</option>
                                    ))}
                                </select>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', marginBottom: '6px' }}>
                                    <label style={styles.label}>P-Value threshold</label>
                                    <span className="text-lg font-black text-[#0055a4]">{config.text_p_threshold}</span>
                                </div>
                                <input
                                    type="range" min="0.01" max="0.20" step="0.01"
                                    className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#0055a4]"
                                    value={config.text_p_threshold}
                                    onChange={(e) => setConfig({ ...config, text_p_threshold: parseFloat(e.target.value) })}
                                />
                                <div className="flex justify-between text-[9px] font-bold text-gray-400 mt-2 uppercase tracking-tighter">
                                    <span className="text-[#ed1c24]">Khắt khe (0.01)</span>
                                    <span>Trung bình</span>
                                    <span>Lỏng lẻo (0.20)</span>
                                </div>
                            </div>

                        </div>
                    </div>
                </div>

                {/* Thông báo kỹ thuật (Technical Notice) */}
                <footer className="bg-[#FFF8E1] border border-[#FFE082] p-4 flex gap-4 items-start shadow-sm">
                    <div className="bg-white p-2 rounded-lg text-[#ed1c24]">
                        <AlertCircle size={20} />
                    </div>
                    <div>
                        <h4 className="text-xs font-bold text-amber-900 uppercase mb-1 tracking-wider">Thông báo vận hành hệ thống</h4>
                        <p className="text-[11px] text-amber-800 leading-relaxed font-medium">
                            Việc thay đổi cấu hình Model Encoder sẽ kích hoạt quy trình tái cấu trúc Feature Space.
                            Toàn bộ dữ liệu đang chờ trong <strong>Bộ đệm (Buffer)</strong> sẽ bị hủy bỏ (Flush) để đảm bảo tính nhất quán của phép thử thống kê tiếp theo.
                        </p>
                    </div>
                </footer>
            </div>
        </GovLayout>
    );
};

export default ConfigPage;