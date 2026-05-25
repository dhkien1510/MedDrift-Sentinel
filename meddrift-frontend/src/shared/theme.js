// Color tokens
export const C = {
    navyDark: '#00336A',
    navy: '#004B9C',
    navyLight: '#005EB8',
    gold: '#D4A017',
    red: '#C0392B',
    white: '#FFFFFF',
    offWhite: '#F2F5F8',
    borderLight: '#C8D8E8',
    textPrimary: '#1A2A3A',
    textSecondary: '#4A5568',
    textMuted: '#718096',
};

// Shared style objects
export const S = {
    page: {
        minHeight: '100vh',
        background: C.offWhite,
        fontFamily: "'Times New Roman', Georgia, serif",
        color: C.textPrimary,
    },
    panel: {
        background: C.white,
        border: `1px solid ${C.borderLight}`,
    },
    panelHeader: {
        background: C.navyDark,
        padding: '10px 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
    },
    panelTitle: {
        fontSize: 13,
        fontWeight: 700,
        color: C.white,
        fontFamily: 'Arial, sans-serif',
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
    },
    sectionTitle: {
        fontSize: 15,
        fontWeight: 700,
        color: C.navyDark,
        borderLeft: `4px solid ${C.gold}`,
        paddingLeft: 12,
        marginBottom: 16,
        fontFamily: 'Arial, sans-serif',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
    },
    tableCell: {
        padding: '8px 12px',
        fontSize: 12,
        fontFamily: 'Arial, sans-serif',
        verticalAlign: 'middle',
    },
    tableRow: (i) => ({
        background: i % 2 === 0 ? C.white : '#F7FAFD',
        borderBottom: `1px solid ${C.borderLight}`,
    }),
};