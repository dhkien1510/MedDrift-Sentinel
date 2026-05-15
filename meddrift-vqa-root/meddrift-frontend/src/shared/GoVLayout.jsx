import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Clock, ChevronRight } from 'lucide-react';
import { C } from './theme';

const NAV_ITEMS = [
    { label: 'Hỏi đáp Y khoa', path: '/chat' },
    { label: 'Cấu hình hệ thống', path: '/config' },
    { label: 'Bảng điều khiển Drift', path: '/dashboard' },
    { label: 'Mô phỏng kịch bản', path: '/test'}
];

const BREADCRUMB_MAP = {
    '/chat': 'Hỏi đáp Y khoa',
    '/config': 'Cấu hình hệ thống',
    '/dashboard': 'Bảng điều khiển Drift',
};

export default function GovLayout({ children, systemStatus = true }) {
    const location = useLocation();
    const navigate = useNavigate();

    return (
        <div style={{ minHeight: '100vh', background: C.offWhite, fontFamily: "'Times New Roman', Georgia, serif" }}>

            {/* Header */}
            <div style={{ background: C.navy, padding: '16px 0', borderBottom: `3px solid ${C.gold}` }}>
                <div style={{ ...inner, gap: 20 }}>

                    <div style={{ flex: 1 }}>

                        <div style={{ fontSize: 22, fontWeight: 700, color: C.white, lineHeight: 1.2, marginBottom: 2 }}>
                            Hệ thống Hỏi đáp & Giám sát Y khoa
                        </div>
                        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.85)', fontFamily: 'Arial, sans-serif' }}>
                            Medical QA & Drift Detection System — HCMUS Project
                        </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                        <div style={{ ...badge(systemStatus), display: 'inline-block' }}>
                            {systemStatus ? '✓ HỆ THỐNG HOẠT ĐỘNG' : '⚠ CÓ SỰ CỐ'}
                        </div>
                    </div>
                </div>
            </div>

            {/* Navbar */}
            <div style={{ background: C.navy, borderBottom: '1px solid rgba(255,255,255,0.15)' }}>
                <div style={{ ...inner, gap: 0 }}>
                    {NAV_ITEMS.map((item) => {
                        const isActive = location.pathname === item.path;
                        return (
                            <div
                                key={item.path}
                                onClick={() => navigate(item.path)}
                                style={{
                                    padding: '10px 18px',
                                    fontSize: 13,
                                    fontFamily: 'Arial, sans-serif',
                                    fontWeight: 600,
                                    borderRight: '1px solid rgba(255,255,255,0.15)',
                                    cursor: 'pointer',
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.05em',
                                    background: isActive ? C.gold : 'transparent',
                                    color: isActive ? C.navyDark : 'rgba(255,255,255,0.9)',
                                }}
                            >
                                {item.label}
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Breadcrumb */}
            <div style={{ background: C.white, borderBottom: `1px solid ${C.borderLight}` }}>
                <div style={{ ...inner, padding: '8px 24px', fontSize: 12, fontFamily: 'Arial, sans-serif', color: C.textSecondary, gap: 6 }}>
                    <span>Trang chủ</span>
                    <ChevronRight size={12} />
                    <span style={{ color: C.navy, fontWeight: 600 }}>
                        {BREADCRUMB_MAP[location.pathname] ?? 'Trang'}
                    </span>
                </div>
            </div>

            {/* Page content */}
            <main style={{ maxWidth: 1200, margin: '0 auto', padding: '24px' }}>
                {children}
            </main>

            {/* Footer */}
            <footer style={{ background: C.navyDark, borderTop: `3px solid ${C.gold}`, padding: '16px 0', marginTop: 32 }}>
                <div style={{ ...inner, justifyContent: 'space-between', fontSize: 12, color: 'rgba(255,255,255,0.7)', fontFamily: 'Arial, sans-serif' }}>
                    <span>23127400 Đinh Hồng Kiên — Medical QA & Drift Detection System · HCMUS Project</span>
                    <span style={{ color: C.gold }}>Phiên bản 1.0.0</span>
                </div>
            </footer>

        </div>
    );
}

// ── helpers ──────────────────────────────────────────────
const inner = {
    maxWidth: 1200,
    margin: '0 auto',
    padding: '0 24px',
    display: 'flex',
    alignItems: 'center',
};

const badge = (ok) => ({
    fontSize: 11,
    fontFamily: 'Arial, sans-serif',
    fontWeight: 700,
    padding: '2px 10px',
    background: ok ? '#27AE60' : '#C0392B',
    color: '#FFFFFF',
    borderRadius: 2,
    letterSpacing: '0.05em',
});

