import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Settings, Save, RefreshCw, Layers, Cpu, Activity } from 'lucide-react';

const ConfigPage = () => {
    const [loading, setLoading] = useState(true);
    const [options, setOptions] = useState({ algorithms: [], encoders: [] });
    const [config, setConfig] = useState({
        image_encoder: '',
        text_encoder: '',
        drift_algorithm: '',
        p_value_threshold: 0.05
    });

    // 1. Load danh sách tùy chọn và cấu hình hiện tại
    useEffect(() => {
        const fetchData = async () => {
            try {
                const [algoRes, configRes] = await Promise.all([
                    axios.get('/api/drift/algorithms'),
                    axios.get('/api/drift/config')
                ]);

                setOptions(prev => ({ ...prev, algorithms: algoRes.data.algorithms }));
                setConfig(configRes.data);
                setLoading(false);
            } catch (err) {
                console.error("Lỗi tải cấu hình:", err);
            }
        };
        fetchData();
    }, []);

    // 2. Xử lý lưu cấu hình mới
    const handleSave = async () => {
        try {
            setLoading(true);
            await axios.post('/api/drift/config/apply', config);
            alert("Cấu hình đã được áp dụng thành công!");
        } catch (err) {
            alert("Lỗi khi áp dụng cấu hình.");
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div className="p-10 text-center">Đang tải cấu hình...</div>;

    return (
        <div className="min-h-screen bg-[#f9f8f6] p-8">
            <div className="max-w-4xl mx-auto">
                <header className="mb-8 flex justify-between items-center">
                    <div>
                        <h1 className="text-2xl font-semibold text-orange-900 flex items-center gap-2">
                            <Settings size={24} /> Cấu hình Sentinel
                        </h1>
                        <p className="text-sm text-gray-500 mt-1">Quản lý Model Registry và Thuật toán Drift</p>
                    </div>
                    <button
                        onClick={handleSave}
                        className="flex items-center gap-2 bg-orange-800 text-white px-6 py-2 rounded-xl hover:bg-orange-900 transition shadow-sm"
                    >
                        <Save size={18} /> Lưu & Áp dụng
                    </button>
                </header>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

                    {/* Cấu hình Image Pipeline */}
                    <section className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
                        <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Layers size={16} /> Image Pipeline
                        </h2>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-gray-700 mb-1">Image Encoder (Feature Extractor)</label>
                                <select
                                    className="w-full bg-gray-50 border border-gray-200 rounded-lg p-2 text-sm outline-none focus:ring-2 focus:ring-orange-200"
                                    value={config.image_encoder}
                                    onChange={(e) => setConfig({ ...config, image_encoder: e.target.value })}
                                >
                                    <option value="clip-vit-b32">CLIP ViT-B/32 (General)</option>
                                    <option value="biomedclip">BiomedCLIP (Medical Special)</option>
                                </select>
                            </div>
                        </div>
                    </section>

                    {/* Cấu hình Text Pipeline */}
                    <section className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
                        <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Cpu size={16} /> Text Pipeline
                        </h2>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-xs font-medium text-gray-700 mb-1">Text Encoder (LLM/BERT)</label>
                                <select
                                    className="w-full bg-gray-50 border border-gray-200 rounded-lg p-2 text-sm outline-none focus:ring-2 focus:ring-orange-200"
                                    value={config.text_encoder}
                                    onChange={(e) => setConfig({ ...config, text_encoder: e.target.value })}
                                >
                                    <option value="biobert">BioBERT (Y sinh)</option>
                                    <option value="pubmedbert">PubMedBERT</option>
                                </select>
                            </div>
                        </div>
                    </section>

                    {/* Cấu hình Drift Algorithm */}
                    <section className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm md:col-span-2">
                        <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                            <Activity size={16} /> Drift Detection Settings
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <label className="block text-xs font-medium text-gray-700 mb-1">Thuật toán giám sát</label>
                                <select
                                    className="w-full bg-gray-50 border border-gray-200 rounded-lg p-2 text-sm outline-none focus:ring-2 focus:ring-orange-200"
                                    value={config.drift_algorithm}
                                    onChange={(e) => setConfig({ ...config, drift_algorithm: e.target.value })}
                                >
                                    {/* Thêm dấu ? sau algorithms */}
                                    {options.algorithms?.map(algo => (
                                        <option key={algo} value={algo}>{algo.toUpperCase()}</option>
                                    ))}
                                </select>
                                <p className="text-[10px] text-gray-400 mt-2 italic">MMD phù hợp cho dữ liệu đa chiều (embeddings).</p>
                            </div>
                            <div>
                                <label className="block text-xs font-medium text-gray-700 mb-1">P-value Threshold: {config.p_value_threshold}</label>
                                <input
                                    type="range" min="0.01" max="0.20" step="0.01"
                                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-800"
                                    value={config.p_value_threshold}
                                    onChange={(e) => setConfig({ ...config, p_value_threshold: parseFloat(e.target.value) })}
                                />
                                <div className="flex justify-between text-[10px] text-gray-400 mt-1">
                                    <span>Khắt khe (0.01)</span>
                                    <span>Lỏng lẻo (0.20)</span>
                                </div>
                            </div>
                        </div>
                    </section>
                </div>

                <footer className="mt-8 p-4 bg-orange-50 rounded-xl border border-orange-100 flex items-start gap-3">
                    <RefreshCw size={18} className="text-orange-800 mt-1" />
                    <div className="text-xs text-orange-900 leading-relaxed">
                        <p className="font-bold">Lưu ý về Software Engineering:</p>
                        Việc áp dụng cấu hình mới sẽ thay đổi cách trích xuất feature từ ảnh/câu hỏi trong buffer tiếp theo. Dữ liệu cũ trong buffer sẽ được flush để đảm bảo tính nhất quán.
                    </div>
                </footer>
            </div>
        </div>
    );
};

export default ConfigPage;