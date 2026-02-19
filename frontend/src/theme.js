import { createTheme } from '@mui/material/styles';

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#D0BCFF',
            contrastText: '#381E72',
        },
        secondary: {
            main: '#CCC2DC',
            contrastText: '#332D41',
        },
        background: {
            default: '#141218',
            paper: '#1D1B20',
        },
        text: {
            primary: '#E6E1E5',
            secondary: '#CAC4D0',
        },
        error: {
            main: '#F2B8B5',
            contrastText: '#601410',
        },
        warning: {
            main: '#FFB74D',
        },
        success: {
            main: '#B6F2B5',
            contrastText: '#00390A',
        },
    },
    typography: {
        fontFamily: '"Google Sans Flex", "Google Sans", "Roboto", "Helvetica", "Arial", sans-serif',
        h1: { fontSize: '2.5rem', fontWeight: 600 },
        h2: { fontSize: '2rem', fontWeight: 500 },
        h3: { fontSize: '1.75rem', fontWeight: 500 },
        h4: { fontSize: '1.5rem', fontWeight: 500 },
        h5: { fontSize: '1.25rem', fontWeight: 500 },
        h6: { fontSize: '1.1rem', fontWeight: 500 },
        subtitle1: { color: '#CAC4D0' },
        body1: { fontSize: '0.95rem' },
        body2: { fontSize: '0.875rem' },
    },
    components: {
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 20,
                    textTransform: 'none',
                    fontWeight: 600,
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    borderRadius: 16,
                    backgroundImage: 'none',
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    borderRadius: 16,
                },
            },
        },
        MuiTab: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    fontWeight: 500,
                    fontSize: '0.95rem',
                },
            },
        },
    },
});

export default theme;
