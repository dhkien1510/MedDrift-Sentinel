import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    AreaChart, Area, ScatterChart, Scatter, ZAxis, Legend, ComposedChart,
} from 'recharts';
import { Activity, AlertTriangle, Database, Clock, ArrowUpRight, ChevronRight, Maximize2 } from 'lucide-react';
import GovLayout from '../shared/GoVLayout';
import { C, S } from '../shared/theme';
import Markdown from 'react-markdown'


const styles = {
    kpiGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 16,
        marginBottom: 24,
    },
    kpiCard: {
        background: C.white,
        border: `1px solid ${C.borderLight}`,
        borderTop: `3px solid ${C.navy}`,
        padding: '16px',
    },
    kpiCardRed: { borderTopColor: C.red },
    kpiCardGold: { borderTopColor: C.gold },
    kpiLabel: {
        fontSize: 11,
        fontFamily: 'Arial, sans-serif',
        fontWeight: 700,
        color: C.textMuted,
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        marginBottom: 6,
    },
    kpiValue: {
        fontSize: 28,
        fontWeight: 700,
        color: C.navyDark,
        lineHeight: 1,
        marginBottom: 8,
        fontFamily: 'Arial, sans-serif',
    },
    kpiValueRed: { color: C.red },
    kpiIconRow: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderTop: `1px dashed ${C.borderLight}`,
        paddingTop: 8,
        marginTop: 4,
    },
    kpiSub: {
        fontSize: 11,
        fontFamily: 'Arial, sans-serif',
        color: C.textMuted,
    },
    chartRow: {
        display: 'grid',
        gridTemplateColumns: '2fr 1fr',
        gap: 16,
        marginBottom: 24,
    },
    panelBody: {
        padding: '16px',
    },
    bufferTrackOuter: {
        width: '100%',
        height: 20,
        background: C.offWhite,
        border: `1px solid ${C.borderLight}`,
        position: 'relative',
        overflow: 'hidden',
    },
    tableRow: (i) => ({
        background: i % 2 === 0 ? C.white : '#F7FAFD',
        borderBottom: `1px solid ${C.borderLight}`,
    }),
    tableCell: {
        padding: '8px 12px',
        fontSize: 12,
        fontFamily: 'Arial, sans-serif',
        verticalAlign: 'middle',
    },
};

/* ─── KPI Card ─── */
const KpiCard = ({ title, value, sub, icon, accent }) => {
    const accentStyle = accent === 'red' ? styles.kpiCardRed : accent === 'gold' ? styles.kpiCardGold : {};
    const valueStyle = accent === 'red' ? { ...styles.kpiValue, ...styles.kpiValueRed } : styles.kpiValue;
    return (
        <div style={{ ...styles.kpiCard, ...accentStyle }}>
            <div style={styles.kpiLabel}>{title}</div>
            <div style={valueStyle}>{value}</div>
            <div style={styles.kpiIconRow}>
                <span style={styles.kpiSub}>{sub}</span>
                <span style={{ color: accent === 'red' ? C.red : C.navy }}>{icon}</span>
            </div>
        </div>
    );
};

/* ─── Custom Tooltip ─── */
const GovTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div style={{ background: C.navyDark, border: `1px solid ${C.gold}`, padding: '8px 14px', fontSize: 12, fontFamily: 'Arial, sans-serif', color: C.white }}>
            <div style={{ color: C.gold, fontWeight: 700, marginBottom: 4 }}>{label}</div>
            {payload.map((p, i) => (
                <div key={i}>{p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</strong></div>
            ))}
        </div>
    );
};

/* ─── Encoder id → short label for buttons ─── */
const encoderShort = (modelId) => {
    if (!modelId || typeof modelId !== 'string') return 'encoder';
    const parts = modelId.split('/');
    return parts[parts.length - 1] || modelId;
};

/* ─── Main Page ─── */
const DashBoardPage = () => {
    const [stats, setStats] = useState({
        total_requests: 0,
        drift_events: 0,
        current_p_value: 1.0,
        buffer_usage: 0,
        buffer_threshold: 100,
        buffer_count: 0,
        agent_analysis: 'Chưa có phân tích nguyên nhân tĩnh',
        severity: '🟢 NORMAL',
        latest_p_image: null,
        latest_p_text: null,
        latest_p_multimodal: null,
        image_p_threshold: 0.05,
        text_p_threshold: 0.05,
        multimodal_p_threshold: 0.05,
        multimodal_enabled: false,
        image_encoder_label: 'Image',
        text_encoder_label: 'Text',
    });
    const [chartData, setChartData] = useState([]);
    const [scatterData, setScatterData] = useState({ image_data: [], text_data: [], multimodal_data: [] });
    const [scatterMode, setScatterMode] = useState('image');
    const [chartPValueMode, setChartPValueMode] = useState('image');
    const [pcaSource, setPcaSource] = useState('auto');
    const [lastUpdated, setLastUpdated] = useState('--:--:--');

    const chartView = useMemo(() => {
        const pKey = chartPValueMode === 'image' ? 'pImage' : chartPValueMode === 'text' ? 'pText' : 'pMultimodal';
        const tKey = chartPValueMode === 'image' ? 'thrImage' : chartPValueMode === 'text' ? 'thrText' : 'thrMultimodal';
        return chartData.map((row) => ({
            time: row.time,
            p: row[pKey],
            thr: row[tKey],
        }));
    }, [chartData, chartPValueMode]);

    const activeThreshold =
        chartPValueMode === 'image'
            ? stats.image_p_threshold
            : chartPValueMode === 'text'
              ? stats.text_p_threshold
              : stats.multimodal_p_threshold;

    const currentPForKpi =
        chartPValueMode === 'image'
            ? stats.latest_p_image
            : chartPValueMode === 'text'
              ? stats.latest_p_text
              : stats.latest_p_multimodal;

    const isDrift =
        currentPForKpi != null &&
        Number.isFinite(Number(currentPForKpi)) &&
        Number(currentPForKpi) < activeThreshold;

    const scatterSeries =
        scatterMode === 'image'
            ? scatterData.image_data
            : scatterMode === 'text'
              ? scatterData.text_data
              : scatterData.multimodal_data || [];

    useEffect(() => {
        const fetchDashboardData = async () => {
            try {
                const statusRes = await axios.get('/api/drift/status');
                const latestReport = statusRes.data.latest_report || {};

                let activeScenario = 'none';
                if (latestReport.simulated && latestReport.scenario) {
                    activeScenario = latestReport.scenario;
                }
                setPcaSource(activeScenario);

                const visEndpoint =
                    activeScenario === 'none'
                        ? '/api/drift/visualization'
                        : `/api/drift/visualization?scenario=${activeScenario}`;

                const [historyRes, visRes, cfgRes] = await Promise.all([
                    axios.get('/api/drift/reports'),
                    axios
                        .get(visEndpoint)
                        .catch(() => ({
                            data: { image_data: [], text_data: [], multimodal_data: [] },
                        })),
                    axios.get('/api/drift/config').catch(() => ({ data: {} })),
                ]);

                const cfg = cfgRes.data || {};
                const bufThr = statusRes.data.buffer_threshold ?? 100;
                const imgThr = typeof cfg.image_p_threshold === 'number' ? cfg.image_p_threshold : 0.05;
                const txtThr = typeof cfg.text_p_threshold === 'number' ? cfg.text_p_threshold : 0.05;
                const mmThr = typeof cfg.multimodal_p_threshold === 'number' ? cfg.multimodal_p_threshold : 0.05;
                const mmEnabled = Boolean(cfg.multimodal_enabled);

                const reportsArray = historyRes.data.reports || [];

                const lip = latestReport.image_drift?.p_value;
                const ltp = latestReport.text_drift?.p_value;
                const mm = latestReport.multimodal_drift;
                const lmp =
                    mm && mm.drift_ran && typeof mm.p_value === 'number' && !Number.isNaN(mm.p_value)
                        ? mm.p_value
                        : null;

                setStats({
                    total_requests: (statusRes.data.total_reports || 0) * bufThr + (statusRes.data.buffer_count || 0),
                    drift_events: reportsArray.filter((d) => Boolean(d.alert)).length,
                    current_p_value: typeof lip === 'number' ? lip : 1.0,
                    buffer_usage: bufThr ? ((statusRes.data.buffer_count || 0) / bufThr) * 100 : 0,
                    buffer_threshold: bufThr,
                    buffer_count: statusRes.data.buffer_count || 0,
                    agent_analysis: latestReport.agent_analysis || 'Chưa có báo cáo phân tích nào từ LLM Agent.',
                    severity: latestReport.severity || '🟢 NORMAL',
                    latest_p_image: typeof lip === 'number' ? lip : null,
                    latest_p_text: typeof ltp === 'number' ? ltp : null,
                    latest_p_multimodal: lmp,
                    image_p_threshold: imgThr,
                    text_p_threshold: txtThr,
                    multimodal_p_threshold: mmThr,
                    multimodal_enabled: mmEnabled,
                    image_encoder_label: encoderShort(cfg.image_encoder),
                    text_encoder_label: encoderShort(cfg.text_encoder),
                });

                const formattedHistory = reportsArray.map((item) => {
                    const mmr = item.multimodal_drift;
                    const mmP =
                        mmr && mmr.drift_ran && typeof mmr.p_value === 'number' && !Number.isNaN(mmr.p_value)
                            ? mmr.p_value
                            : null;
                    return {
                        time: new Date(item.timestamp).toLocaleTimeString('vi-VN', {
                            hour: '2-digit',
                            minute: '2-digit',
                        }),
                        pImage: typeof item.image_drift?.p_value === 'number' ? item.image_drift.p_value : null,
                        pText: typeof item.text_drift?.p_value === 'number' ? item.text_drift.p_value : null,
                        pMultimodal: mmP,
                        thrImage: imgThr,
                        thrText: txtThr,
                        thrMultimodal: mmThr,
                    };
                });
                setChartData(formattedHistory);
                setScatterData({
                    image_data: visRes.data.image_data || [],
                    text_data: visRes.data.text_data || [],
                    multimodal_data: visRes.data.multimodal_data || [],
                });
            } catch (err) {
                console.error('Lỗi tải dữ liệu Dashboard:', err);
            }
            setLastUpdated(new Date().toLocaleTimeString('vi-VN'));
        };

        fetchDashboardData();
        const timer = setInterval(fetchDashboardData, 10000);
        return () => clearInterval(timer);
    }, []);

    useEffect(() => {
        if (!stats.multimodal_enabled && scatterMode === 'multimodal') {
            setScatterMode('image');
        }
        if (!stats.multimodal_enabled && chartPValueMode === 'multimodal') {
            setChartPValueMode('image');
        }
    }, [stats.multimodal_enabled, scatterMode, chartPValueMode]);

    const bufferPct = Math.min(100, Math.max(0, stats.buffer_usage));

    return (
        <GovLayout systemStatus={!isDrift}>


            {/* Section: KPI */}
            <div style={S.sectionTitle}>
                <Activity size={16} color={C.navy} />
                Chỉ số thống kê tổng hợp
            </div>
            <div style={styles.kpiGrid}>
                <KpiCard
                    title="Tổng số yêu cầu"
                    value={stats.total_requests.toLocaleString('vi-VN')}
                    sub="Dữ liệu tích lũy"

                    accent="navy"
                />
                <KpiCard
                    title="Sự kiện Drift"
                    value={stats.drift_events}
                    sub="Lần phát hiện bất thường"

                    accent="red"
                />
                <KpiCard
                    title="P-Value hiện tại"
                    value={
                        currentPForKpi != null && Number.isFinite(Number(currentPForKpi))
                            ? Number(currentPForKpi).toFixed(4)
                            : '—'
                    }
                    sub={
                        chartPValueMode === 'multimodal' && (currentPForKpi == null || !Number.isFinite(Number(currentPForKpi)))
                            ? 'Chưa có lô đa phương thức (bật multimodal + build reference)'
                            : isDrift
                              ? `Dưới ngưỡng cấu hình (${activeThreshold})`
                              : `So với ngưỡng ${activeThreshold}`
                    }
                    accent={isDrift ? 'red' : 'navy'}
                />
                <KpiCard
                    title="Buffer Usage"
                    value={`${bufferPct.toFixed(0)}%`}
                    sub="Độ bão hòa bộ đệm"

                    accent={bufferPct > 80 ? 'red' : 'gold'}
                />
            </div>

            {/* Section: Charts */}
            <div style={S.sectionTitle}>
                <Activity size={16} color={C.navy} />
                Biểu đồ phân tích
            </div>
            <div style={styles.chartRow}>
                {/* P-Value chart */}
                <div style={S.panel}>
                    <div style={S.panelHeader}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 8, width: '100%' }}>
                            <div style={S.panelTitle}>
                                <Activity size={14} aria-hidden="true" />
                                Xu hướng P-Value theo thời gian thực
                            </div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
                                <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.75)', fontFamily: 'Arial', marginRight: 4 }}>
                                    Chế độ:
                                </span>
                                {[
                                    { id: 'image', label: `Ảnh (${stats.image_encoder_label})` },
                                    { id: 'text', label: `Văn bản (${stats.text_encoder_label})` },
                                    { id: 'multimodal', label: 'Đa phương thức' },
                                ].map((b) => {
                                    const disabled = b.id === 'multimodal' && !stats.multimodal_enabled;
                                    return (
                                    <button
                                        key={b.id}
                                        type="button"
                                        disabled={disabled}
                                        title={disabled ? 'Bật multimodal.enabled và build PCA trong drift_config' : ''}
                                        onClick={() => !disabled && setChartPValueMode(b.id)}
                                        style={{
                                            padding: '4px 10px',
                                            fontSize: 10,
                                            fontWeight: 'bold',
                                            border: '1px solid rgba(255,255,255,0.5)',
                                            background: chartPValueMode === b.id ? C.white : 'transparent',
                                            color: chartPValueMode === b.id ? C.navy : C.white,
                                            cursor: disabled ? 'not-allowed' : 'pointer',
                                            borderRadius: 4,
                                            fontFamily: 'Arial',
                                            opacity: disabled ? 0.45 : 1,
                                        }}
                                    >
                                        {b.label}
                                    </button>
                                    );
                                })}
                            </div>
                        </div>
                        <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.6)', fontFamily: 'Arial', marginTop: 6, display: 'block' }}>
                            Ngưỡng cảnh báo (theo cấu hình): {activeThreshold}
                        </span>
                    </div>
                    <div style={styles.panelBody}>
                        <div style={{ height: 300 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart data={chartView} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="pValGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor={C.navy} stopOpacity={0.15} />
                                            <stop offset="95%" stopColor={C.navy} stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke={C.borderLight} vertical={false} />
                                    <XAxis dataKey="time" tick={{ fontSize: 11, fontFamily: 'Arial', fill: C.textMuted }} axisLine={false} tickLine={false} dy={6} />
                                    <YAxis domain={[0, 1]} tick={{ fontSize: 11, fontFamily: 'Arial', fill: C.textMuted }} axisLine={false} tickLine={false} />
                                    <Tooltip content={<GovTooltip />} />
                                    <Area
                                        type="monotone"
                                        dataKey="p"
                                        name="P-Value"
                                        stroke={C.navy}
                                        strokeWidth={2}
                                        fill="url(#pValGrad)"
                                        dot={false}
                                        connectNulls={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="thr"
                                        name="Ngưỡng"
                                        stroke={C.red}
                                        strokeDasharray="6 3"
                                        strokeWidth={1.5}
                                        dot={false}
                                        connectNulls={false}
                                    />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>
                        <div style={{ display: 'flex', gap: 20, marginTop: 8, fontSize: 11, fontFamily: 'Arial', color: C.textMuted, flexWrap: 'wrap' }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span style={{ display: 'inline-block', width: 24, height: 2, background: C.navy }} />
                                P-Value ({chartPValueMode})
                            </span>
                            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span style={{ display: 'inline-block', width: 24, height: 2, background: C.red, borderTop: '2px dashed' }} />
                                Ngưỡng ({activeThreshold})
                            </span>
                        </div>
                    </div>
                </div>

                {/* Buffer panel */}
                <div style={S.panel}>
                    <div style={S.panelHeader}>
                        <div style={S.panelTitle}>

                            Trạng thái Buffer
                        </div>
                    </div>
                    <div style={{ ...styles.panelBody, display: 'flex', flexDirection: 'column', gap: 20 }}>
                        {/* Big number */}
                        <div style={{ textAlign: 'center', padding: '16px 0' }}>
                            <div style={{ fontSize: 52, fontWeight: 700, color: bufferPct > 80 ? C.red : C.navyDark, fontFamily: 'Arial', lineHeight: 1 }}>
                                {bufferPct.toFixed(0)}%
                            </div>
                            <div style={{ fontSize: 12, color: C.textMuted, fontFamily: 'Arial', marginTop: 6, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                                Độ bão hòa
                            </div>
                        </div>

                        {/* Progress bar */}
                        <div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, fontFamily: 'Arial', color: C.textMuted, marginBottom: 6 }}>
                                <span>0%</span><span>50%</span><span>100%</span>
                            </div>
                            <div style={styles.bufferTrackOuter}>
                                <div style={{
                                    position: 'absolute', left: 0, top: 0, bottom: 0,
                                    width: `${bufferPct}%`,
                                    background: bufferPct > 80 ? C.red : C.navy,
                                    transition: 'width 1s ease',
                                }} />
                                {/* Threshold marker at 100% */}
                                <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: 2, background: C.gold }} />
                            </div>
                        </div>

                        {/* Info rows */}
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'Arial' }}>
                            <tbody>
                                {[
                                    ['Ngưỡng flush', '100%'],
                                    ['Cập nhật mỗi', '10 giây'],
                                    [
                                        'Phân tích',
                                        bufferPct >= 100 ? 'Đang thực hiện' : 'Chờ đầy buffer',
                                    ],
                                    ['Buffer / ngưỡng', `${stats.buffer_count}/${stats.buffer_threshold}`],
                                ].map(([k, v], i) => (
                                    <tr key={i} style={styles.tableRow(i)}>
                                        <td style={{ ...styles.tableCell, color: C.textMuted }}>{k}</td>
                                        <td style={{ ...styles.tableCell, fontWeight: 700, color: C.textPrimary, textAlign: 'right' }}>{v}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>

                        <div style={{ background: '#FFF8E1', border: `1px solid ${C.gold}`, padding: '10px 12px', fontSize: 11, fontFamily: 'Arial', color: '#7A5C00', lineHeight: 1.6 }}>
                            <strong>Lưu ý:</strong> Hệ thống flush buffer khi đạt đủ {stats.buffer_threshold} mẫu (theo cấu hình).
                        </div>
                    </div>
                </div>
            </div>

            {/* Section: Scatter Plot (Visualization) */}
            <div style={{ ...S.sectionTitle, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <Maximize2 size={16} color={C.navy} />
                    Ánh xạ Không gian dữ liệu (Dimensionality Reduction)
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <div style={{ fontSize: '11px', color: C.textMuted, marginRight: 8 }}>
                        Nguồn PCA: <strong>{pcaSource === 'none' ? 'Buffer Mặc định' : `Mô phỏng (${pcaSource})`}</strong>
                    </div>

                    <button
                        onClick={() => setScatterMode('image')}
                        style={{
                            padding: '6px 12px',
                            fontSize: '11px',
                            fontWeight: 'bold',
                            border: `1px solid ${C.navy}`,
                            background: scatterMode === 'image' ? C.navy : 'transparent',
                            color: scatterMode === 'image' ? C.white : C.navy,
                            cursor: 'pointer',
                            borderRadius: '4px',
                        }}
                    >
                        Ảnh ({stats.image_encoder_label})
                    </button>
                    <button
                        onClick={() => setScatterMode('text')}
                        style={{
                            padding: '6px 12px',
                            fontSize: '11px',
                            fontWeight: 'bold',
                            border: `1px solid ${C.navy}`,
                            background: scatterMode === 'text' ? C.navy : 'transparent',
                            color: scatterMode === 'text' ? C.white : C.navy,
                            cursor: 'pointer',
                            borderRadius: '4px',
                        }}
                    >
                        Văn bản ({stats.text_encoder_label})
                    </button>
                    {stats.multimodal_enabled ? (
                        <button
                            onClick={() => setScatterMode('multimodal')}
                            style={{
                                padding: '6px 12px',
                                fontSize: '11px',
                                fontWeight: 'bold',
                                border: `1px solid ${C.navy}`,
                                background: scatterMode === 'multimodal' ? C.navy : 'transparent',
                                color: scatterMode === 'multimodal' ? C.white : C.navy,
                                cursor: 'pointer',
                                borderRadius: '4px',
                            }}
                        >
                            Đa phương thức (PCA+MMD joint)
                        </button>
                    ) : null}
                </div>
            </div>
            <div style={{ ...S.panel, marginBottom: 24 }}>
                <div style={S.panelHeader}>
                    <div style={S.panelTitle}>
                        <Activity size={14} aria-hidden="true" />
                        Biểu đồ phân tán (
                        {scatterMode === 'image'
                            ? `Ảnh — ${stats.image_encoder_label}`
                            : scatterMode === 'text'
                              ? `Văn bản — ${stats.text_encoder_label}`
                              : 'Đa phương thức — joint sau PCA'}
                        )
                    </div>
                </div>
                <div style={styles.panelBody}>
                    {scatterSeries.length > 0 ? (
                        <div style={{ height: 400, width: '100%' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis type="number" dataKey="x" name="PCA 1" tick={{ fontSize: 11, fontFamily: 'Arial' }} />
                                    <YAxis type="number" dataKey="y" name="PCA 2" tick={{ fontSize: 11, fontFamily: 'Arial' }} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: 12, fontFamily: 'Arial' }} />
                                    <Legend wrapperStyle={{ fontSize: 12, fontFamily: 'Arial' }} />
                                    <Scatter
                                        name="Dữ liệu tham chiếu (chuẩn)"
                                        data={scatterSeries.filter((d) => d.type === 'Reference')}
                                        fill="#94a3b8"
                                    />
                                    <Scatter
                                        name={
                                            pcaSource === 'none'
                                                ? 'Dữ liệu đang phân tích (Current)'
                                                : `Dữ liệu Test mô phỏng (${pcaSource})`
                                        }
                                        data={scatterSeries.filter((d) => d.type === 'Current' || d.type === 'Simulation')}
                                        fill={pcaSource === 'none' ? '#ed1c24' : '#0055a4'}
                                        shape="square"
                                    />
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div style={{ padding: 40, textAlign: 'center', color: C.textMuted, fontSize: 13, fontStyle: 'italic' }}>
                            {scatterMode === 'multimodal' && !stats.multimodal_enabled
                                ? 'Đa phương thức đang tắt trong cấu hình hoặc chưa có PCA/tham chiếu — bật multimodal và chạy build_multimodal_reference.'
                                : scatterMode === 'multimodal'
                                  ? 'Chưa có đủ dữ liệu joint (buffer trống hoặc thiếu artifact PCA).'
                                  : 'Chưa có dữ liệu trong buffer / kịch bản cho chế độ này.'}
                        </div>
                    )}
                </div>
            </div>

            {/* Section: Agent Analysis */}
            <div style={S.sectionTitle}>
                <Database size={16} color={C.navy} />
                Báo cáo chẩn đoán sự cố phân phối (LangChain Agent)
            </div>
            <div style={{ ...S.panel, marginBottom: 40 }}>
                <div style={S.panelHeader}>
                    <div style={S.panelTitle}>
                        <Activity size={14} aria-hidden="true" />
                        Trạng thái AI: {stats.severity}
                    </div>
                </div>
                <div style={{ ...styles.panelBody, background: '#F8FAFC' }}>
                    <div style={{ 
                        fontSize: 13, 
                        lineHeight: 1.6, 
                        color: C.textPrimary, 
                        whiteSpace: 'pre-wrap',
                        fontFamily: 'Arial, sans-serif'
                    }}>
                        <Markdown>{stats.agent_analysis}</Markdown>
                    </div>
                </div>
            </div>

        </GovLayout>




    );
};

export default DashBoardPage;