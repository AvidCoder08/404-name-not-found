import { Paper, Typography, Box, Grid } from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import GroupsIcon from '@mui/icons-material/Groups';

const StatCard = ({ title, value, icon, color }) => (
    <Paper
        elevation={2}
        sx={{
            p: 3,
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            background: (theme) => `linear-gradient(135deg, ${theme.palette.background.paper} 0%, ${color}22 100%)`,
            height: '100%'
        }}
    >
        <Box sx={{ p: 1.5, borderRadius: '50%', backgroundColor: `${color}33`, color: color }}>
            {icon}
        </Box>
        <Box>
            <Typography variant="body2" color="text.secondary">
                {title}
            </Typography>
            <Typography variant="h4" fontWeight="bold">
                {value}
            </Typography>
        </Box>
    </Paper>
);

const ScoreCard = ({ summary }) => {
    if (!summary) return null;

    return (
        <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} md={4}>
                <StatCard
                    title="Total Transactions"
                    value={summary.total_transactions}
                    icon={<AccountBalanceWalletIcon />}
                    color="#D0BCFF" // Primary
                />
            </Grid>
            <Grid item xs={12} md={4}>
                <StatCard
                    title="Flagged Accounts"
                    value={summary.accounts_flagged}
                    icon={<WarningAmberIcon />}
                    color="#F2B8B5" // Error
                />
            </Grid>
            <Grid item xs={12} md={4}>
                <StatCard
                    title="Fraud Rings Detected"
                    value={summary.fraud_rings_detected}
                    icon={<GroupsIcon />}
                    color="#B6F2B5" // Success (or Warning for detection) -> using green for "Detection Success" or maybe Orange?
                // Let's use orange for Warning/Alert
                />
            </Grid>
        </Grid>
    );
};

export default ScoreCard;
