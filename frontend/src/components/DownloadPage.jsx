import { Box, Paper, Typography, Button, Divider } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import DataObjectIcon from '@mui/icons-material/DataObject';
import TableChartIcon from '@mui/icons-material/TableChart';
import { motion } from 'framer-motion';

const DownloadPage = ({ data }) => {
    if (!data) return null;

    const downloadJSON = () => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'analysis_results.json';
        a.click();
        URL.revokeObjectURL(url);
    };

    const downloadCSV = () => {
        // Build CSV from suspicion scores
        const rows = [['Account ID', 'Suspicion Score', 'Risk Level']];
        data.suspicious_accounts
            .forEach(({ account_id, suspicion_score }) => {
                const risk = suspicion_score >= 90 ? 'Critical' : suspicion_score >= 70 ? 'High' : suspicion_score >= 50 ? 'Medium' : 'Low';
                rows.push([account_id, suspicion_score.toFixed(2), risk]);
            });

        const csv = rows.map(r => r.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'suspicion_scores.csv';
        a.click();
        URL.revokeObjectURL(url);
    };

    const downloadRingsCSV = () => {
        const rows = [['Ring ID', 'Pattern Type', 'Member Count', 'Member Accounts', 'Risk Score']];
        data.fraud_rings.forEach(ring => {
            rows.push([
                ring.ring_id,
                ring.pattern_type,
                ring.member_accounts.length,
                `"${ring.member_accounts.join('; ')}"`,
                ring.risk_score,
            ]);
        });

        const csv = rows.map(r => r.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'fraud_rings.csv';
        a.click();
        URL.revokeObjectURL(url);
    };

    const cards = [
        {
            title: 'Full Analysis Report (JSON)',
            description: 'Complete analysis data including suspicion scores, fraud rings, and summary statistics.',
            icon: <DataObjectIcon sx={{ fontSize: 36 }} />,
            color: '#D0BCFF',
            onClick: downloadJSON,
            buttonText: 'Download JSON',
        },
        {
            title: 'Suspicion Scores (CSV)',
            description: 'All account suspicion scores with risk levels for spreadsheet analysis.',
            icon: <TableChartIcon sx={{ fontSize: 36 }} />,
            color: '#F2B8B5',
            onClick: downloadCSV,
            buttonText: 'Download CSV',
        },
        {
            title: 'Fraud Rings (CSV)',
            description: 'Fraud ring details with member accounts and pattern types.',
            icon: <TableChartIcon sx={{ fontSize: 36 }} />,
            color: '#B6F2B5',
            onClick: downloadRingsCSV,
            buttonText: 'Download CSV',
        },
    ];

    return (
        <Box
            component={motion.div}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
        >
            <Typography variant="h5" fontWeight={600} gutterBottom>
                Export Results
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
                Download analysis results in various formats for reporting or further analysis.
            </Typography>

            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 3 }}>
                {cards.map((card, index) => (
                    <Paper
                        key={index}
                        component={motion.div}
                        whileHover={{ scale: 1.02, y: -4 }}
                        transition={{ duration: 0.2 }}
                        elevation={0}
                        sx={{
                            p: 4,
                            border: `1px solid ${card.color}30`,
                            background: `linear-gradient(160deg, transparent 0%, ${card.color}08 100%)`,
                            display: 'flex',
                            flexDirection: 'column',
                            gap: 2,
                            cursor: 'default',
                        }}
                    >
                        <Box sx={{ color: card.color }}>{card.icon}</Box>
                        <Typography variant="h6" fontWeight={600}>{card.title}</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
                            {card.description}
                        </Typography>
                        <Button
                            variant="contained"
                            startIcon={<DownloadIcon />}
                            onClick={card.onClick}
                            sx={{
                                mt: 1,
                                bgcolor: `${card.color}20`,
                                color: card.color,
                                '&:hover': { bgcolor: `${card.color}35` },
                            }}
                        >
                            {card.buttonText}
                        </Button>
                    </Paper>
                ))}
            </Box>
        </Box>
    );
};

export default DownloadPage;
