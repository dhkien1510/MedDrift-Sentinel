import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    AreaChart, Area, BarChart, Bar, Cell
} from 'recharts';
import { Activity, AlertTriangle, Database, Clock, ArrowUpRight } from 'lucide-react';

const DashBoardPage = () => {
    const [stats, setStats] = useState({
        total_requests: 0,
        drift_events: 0,
        current_p_value: 1.0,
        buffer_usage: 0
    });
    const [chartData, setChartData] = useState([]);

    // Fetch dữ liệu thống kê và lịch sử drift
    useEffect(() => {
        const fetchDashboardData = async () => {
            try {
                const [statusRes, historyRes] = await Promise.all([
                    axios.get('/api/drift/status'),
                    axios.get('/api/drift/history') // Bạn cần bổ sung endpoint này ở FastAPI
                ]);

                setStats({
                    total_requests: statusRes.data.total_collected || 0,
                    drift_events: historyRes.data.filter(d => d.is_drift).length,
                    current_p_value: statusRes.data.last_p_value || 1.0,
                    buffer_usage: (statusRes.data.buffer_count / statusRes.data.buffer_threshold) * 100
                });

                // Định dạng dữ liệu cho biểu đồ Line Chart
                const formattedHistory = historyRes.data.map((item, index) => ({
                    time: new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                    pValue: item.p_value,
                    threshold: 0.05 // Ngưỡng mặc định
                }));
                setChartData(formattedHistory);
            } catch (err) {
                console.error("Lỗi tải dữ liệu Dashboard:", err);
            }
        };

        fetchDashboardData();
        const timer = setInterval(fetchDashboardData, 10000); // Cập nhật mỗi 10 giây
        return () => clearInterval(timer);
    }, []);

    return (
        <div className="min-h-screen bg-[#f9f8f6] p-8">
            <div className="max-w-7xl mx-auto space-y-8">

                {/* Header section */}
                <header className="flex justify-between items-end">
                    <div>
                        <h1 className="text-3xl font-bold text-orange-900 tracking-tight">Hệ thống Giám sát Drift</h1>
                        <p className="text-gray-500 mt-2 font-medium">Báo cáo phân phối dữ liệu thời gian thực — HCMUS Project</p>
                    </div>
                    <div className="flex gap-3 text-xs font-bold uppercase tracking-widest text-gray-400">
                        <span className="flex items-center gap-1"><Clock size={14} /> Cập nhật lần cuối: {new Date().toLocaleTimeString()}</span>
                    </div>
                </header>

                {/* Thống kê nhanh (KPI Cards) */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <StatCard title="Tổng yêu cầu" value={stats.total_requests} icon={<Database className="text-blue-600" />} color="blue" />
                    <StatCard title="Sự kiện Drift" value={stats.drift_events} icon={<AlertTriangle className="text-red-600" />} color="red" />
                    <StatCard title="P-Value hiện tại" value={stats.current_p_value.toFixed(4)} icon={<Activity className="text-emerald-600" />} color="emerald" />
                    <StatCard title="Buffer Usage" value={`${stats.buffer_usage.toFixed(0)}%`} icon={<ArrowUpRight className="text-orange-600" />} color="orange" />
                </div>

                {/* Biểu đồ chính: P-Value Timeline */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    <div className="lg:col-span-2 bg-white p-6 rounded-3xl shadow-sm border border-gray-100">
                        <h3 className="text-lg font-bold text-gray-800 mb-6 flex items-center gap-2">
                            <Activity size={20} className="text-orange-800" /> Xu hướng P-Value (Drift Trend)
                        </h3>
                        <div className="h-[350px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={chartData}>
                                    <defs>
                                        <linearGradient id="colorP" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#9a3412" stopOpacity={0.1} />
                                            <stop offset="95%" stopColor="#9a3412" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                                    <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#9ca3af' }} dy={10} />
                                    <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#9ca3af' }} />
                                    <Tooltip
                                        contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                                    />
                                    <Area type="monotone" dataKey="pValue" stroke="#9a3412" strokeWidth={3} fillOpacity={1} fill="url(#colorP)" />
                                    {/* Ngưỡng cảnh báo drift (0.05) */}
                                    <Line type="monotone" dataKey="threshold" stroke="#ef4444" strokeDasharray="5 5" dot={false} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Biểu đồ phụ: Buffer Saturation */}
                    <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100 flex flex-col">
                        <h3 className="text-lg font-bold text-gray-800 mb-6">Trạng thái Buffer</h3>
                        <div className="flex-1 flex flex-col justify-center items-center relative">
                            <div className="text-4xl font-black text-orange-900">{stats.buffer_usage.toFixed(0)}%</div>
                            <div className="text-sm text-gray-400 mt-2 uppercase font-bold">Độ bão hòa</div>

                            {/* Thanh Progress bar dạng đứng */}
                            <div className="mt-8 w-16 h-48 bg-gray-100 rounded-full overflow-hidden relative border border-gray-200">
                                <div
                                    className="absolute bottom-0 w-full bg-orange-800 transition-all duration-1000"
                                    style={{ height: `${stats.buffer_usage}%` }}
                                >
                                    <div className="w-full h-full opacity-20 bg-white animate-pulse" />
                                </div>
                            </div>
                        </div>
                        <p className="mt-6 text-xs text-center text-gray-400 leading-relaxed">
                            Hệ thống sẽ thực hiện phân tích thống kê (flush) khi buffer đạt 100%.
                        </p>
                    </div>
                </div>

            </div>
        </div>
    );
};

// Component con cho các thẻ chỉ số
const StatCard = ({ title, value, icon, color }) => (
    <div className="bg-white p-6 rounded-3xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
        <div className={`w-10 h-10 rounded-2xl bg-${color}-50 flex items-center justify-center mb-4`}>
            {icon}
        </div>
        <div className="text-sm font-bold text-gray-400 uppercase tracking-wider">{title}</div>
        <div className="text-2xl font-black text-gray-800 mt-1">{value}</div>
    </div>
);

export default DashBoardPage;