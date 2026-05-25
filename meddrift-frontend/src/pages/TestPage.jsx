import React, { useState } from 'react';
import axios from 'axios';
import { Play, Activity, CheckCircle, Database } from 'lucide-react';
import GovLayout from '../shared/GoVLayout';
import { C, S } from '../shared/theme';

const SCENARIOS = [
    { value: 'level1_mild', label: 'Image/Text Level 1 (Mild)', desc: 'Biến đổi nhẹ hình ảnh (độ sáng, nhiễu), thay thế từ viết tắt y khoa.' },
    { value: 'level2_moderate', label: 'Image/Text Level 2 (Moderate)', desc: 'Làm mờ, nhiễu mức độ vừa. Câu hỏi đa ngôn ngữ hoặc từ ngữ phức tạp.' },
    { value: 'level3_severe', label: 'Image/Text Level 3 (Severe)', desc: 'Ảnh nhiễu nặng, che khuất. Văn bản ngẫu nhiên hoặc hỏi hoàn toàn lạc đề.' },
    { value: 'no_drift', label: 'No Drift (Control)', desc: 'Dữ liệu y khoa nguyên bản.' }
];

const TestPage = () => {
    const [scenario, setScenario] = useState('level1_mild');
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');

    const handleRunSimulation = async () => {
        try {
            setLoading(true);
            setMessage('');
            await axios.post(`/api/drift/simulate_scenario?scenario_name=${scenario}`);
            setMessage('Mô phỏng đã được khởi chạy trong nền. Truy cập Dashboard (Bảng điều khiển) để xem quá trình xử lý theo thời gian thực!');
        } catch (err) {
            setMessage('Lỗi: Không thể chạy mô phỏng!');
        } finally {
            setLoading(false);
        }
    };

    return (
        <GovLayout>
            <div className="max-w-4xl mx-auto space-y-6">
                <header className="flex justify-between items-end border-b-2 border-gray-100 pb-4">
                    <div>
                        <div className="flex items-center gap-2 text-[#0055a4] mb-1">
                            <Activity size={20} />
                            <h1 className="text-xl font-black uppercase tracking-tight">Trình mô phỏng Kịch bản (Stress Test)</h1>
                        </div>
                        <p className="text-sm text-gray-500 font-medium">Bơm trực tiếp dữ liệu thử nghiệm vào Pipeline để kiểm tra khả năng phát hiện Drift</p>
                    </div>
                </header>

                <div style={S.panel}>
                    <div style={S.panelHeader}>
                        <div style={S.panelTitle}>Cấu hình Nạp giả lập (Simulation)</div>
                    </div>
                    <div className="p-6">
                        <div className="mb-6">
                            <label className="block text-[11px] font-bold text-gray-500 uppercase tracking-wider mb-2">Chọn kịch bản Drift</label>
                            
                            <div className="space-y-3">
                                {SCENARIOS.map(sc => (
                                    <label key={sc.value} className={`flex items-start p-3 border rounded cursor-pointer transition-colors ${scenario === sc.value ? 'bg-[#F0F7FF] border-[#0055a4]' : 'hover:bg-gray-50'}`}>
                                        <input 
                                            type="radio" 
                                            name="scenario" 
                                            value={sc.value} 
                                            checked={scenario === sc.value}
                                            onChange={(e) => setScenario(e.target.value)}
                                            className="mt-1 mr-3 accent-[#0055a4]"
                                        />
                                        <div>
                                            <div className="text-sm font-bold text-gray-800">{sc.label}</div>
                                            <div className="text-xs text-gray-500">{sc.desc}</div>
                                        </div>
                                    </label>
                                ))}
                            </div>
                        </div>

                        <div className="flex items-center justify-between">
                            <button 
                                onClick={handleRunSimulation} 
                                disabled={loading}
                                className={`flex items-center gap-2 px-6 py-2.5 rounded font-bold uppercase tracking-wider text-sm text-white ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-[#0055a4] hover:bg-[#004485] shadow-md transition-all active:scale-95'}`}
                            >
                                <Play size={16} /> {loading ? 'Đang gọi Backend...' : 'Bắt đầu giả lập'}
                            </button>
                        </div>

                        {message && (
                            <div className={`mt-4 p-3 rounded flex items-center gap-2 text-sm font-medium ${message.includes('Lỗi') ? 'bg-red-50 text-red-700' : 'bg-green-50 text-green-700'}`}>
                                <CheckCircle size={16} /> {message}
                            </div>
                        )}
                        
                        <div className="mt-8 pt-4 border-t border-gray-100 flex items-start gap-4">
                            <Database className="text-gray-400" size={24} />
                            <div>
                                <h4 className="text-xs font-bold text-gray-600 uppercase mb-1">Cơ chế hoạt động</h4>
                                <p className="text-[11px] text-gray-500 leading-relaxed font-medium">
                                    Khi kích hoạt, hệ thống sẽ chèn trực tiếp 500 mẫu (embeddings) của kịch bản bạn chọn từ <strong>/data/drift_scenarios/</strong> vào Pipeline nhận diện. Dữ liệu này sẽ được tự động chia nhỏ theo <strong>BUFFER_THRESHOLD</strong> đang được cấu hình tại trang Cài đặt. Kết quả sẽ được in ra tại Dashboard liên tục.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </GovLayout>
    );
};

export default TestPage;