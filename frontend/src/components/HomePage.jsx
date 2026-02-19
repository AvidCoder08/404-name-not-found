import { Box, Grid, Typography, Paper, Divider, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import GroupsIcon from '@mui/icons-material/Groups';
import SecurityIcon from '@mui/icons-material/Security';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

const StatCard = ({ title, value, icon, color, subtitle }) => (
    <Paper
        component={motion.div}
        whileHover={{ scale: 1.02, y: -4 }}
        transition={{ duration: 0.2 }}
        elevation={0}
        sx={{
            p: 3,
            display: 'flex',
            alignItems: 'center',
            gap: 2.5,
            background: (theme) => `linear-gradient(135deg, ${theme.palette.background.paper} 0%, ${color}15 100%)`,
            border: `1px solid ${color}30`,
            height: '100%',
            cursor: 'default',
        }}
    >
        <Box sx={{
            p: 2,
            borderRadius: 3,
            backgroundColor: `${color}20`,
            color: color,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
        }}>
            {icon}
        </Box>
        <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                {title}
            </Typography>
            <Typography variant="h3" fontWeight="bold" sx={{ lineHeight: 1 }}>
                {value}
            </Typography>
            {subtitle && (
                <Typography variant="caption" color="text.secondary">{subtitle}</Typography>
            )}
        </Box>
    </Paper>
);

const RingSummaryCard = ({ ring }) => {
    const patternColors = {
        cycle: '#34A853',
        fan_out: '#4285F4',
        fan_in: '#4285F4',
        fan_in_smurfing: '#4285F4',
        fan_out_smurfing: '#4285F4',
        layered_shell: '#FBBC05',
        layered_shell_network: '#FBBC05',
    };
    const color = patternColors[ring.pattern_type] || '#9AA0A6';

    return (
        <Paper
            elevation={0}
            sx={{
                p: 2.5,
                border: `1px solid ${color}40`,
                background: `linear-gradient(135deg, transparent 0%, ${color}08 100%)`,
            }}
        >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="subtitle2" fontWeight="bold">{ring.ring_id}</Typography>
                <Chip
                    label={ring.pattern_type.replace('_', ' ')}
                    size="small"
                    sx={{ bgcolor: `${color}25`, color: color, fontWeight: 600 }}
                />
            </Box>
            <Typography variant="body2" color="text.secondary">
                {ring.member_accounts.length} accounts involved
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mt: 1, flexWrap: 'wrap' }}>
                {ring.member_accounts.slice(0, 5).map((acc) => (
                    <Chip key={acc} label={acc} size="small" variant="outlined" sx={{ borderColor: 'rgba(255,255,255,0.15)' }} />
                ))}
                {ring.member_accounts.length > 5 && (
                    <Chip label={`+${ring.member_accounts.length - 5} more`} size="small" variant="outlined" sx={{ borderColor: 'rgba(255,255,255,0.15)' }} />
                )}
            </Box>
        </Paper>
    );
};

const HomePage = ({ data }) => {
    if (!data) return null;

    const { summary, suspicious_accounts, fraud_rings } = data;

    // Calculate stats from suspicious_accounts list
    const highRiskCount = suspicious_accounts.filter(acc => acc.suspicion_score >= 70).length;
    const topSuspicious = suspicious_accounts.slice(0, 5);

    // Calculate fraud type stats
    const cycleFraudCount = fraud_rings.filter(r => r.pattern_type === 'cycle').length;
    const smurfFraudCount = fraud_rings.filter(r => r.pattern_type.includes('fan_out') || r.pattern_type.includes('fan_in')).length;
    const shellFraudCount = fraud_rings.filter(r => r.pattern_type.includes('layered_shell')).length;

    return (
        <Box
            component={motion.div}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
        >
            {/* Summary Stats */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6} lg={4}>
                    <StatCard
                        title="Accounts Analyzed"
                        value={(summary.total_accounts_analyzed || summary.total_transactions || 0).toLocaleString()}
                        icon={<AccountBalanceWalletIcon sx={{ fontSize: 28 }} />}
                        color="#4285F4"
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={4}>
                    <StatCard
                        title="Flagged Accounts"
                        value={summary.suspicious_accounts_flagged || summary.accounts_flagged || 0}
                        icon={<WarningAmberIcon sx={{ fontSize: 28 }} />}
                        color="#EA4335"
                        subtitle="Suspicion > 50%"
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={4}>
                    <StatCard
                        title="High Risk Accounts"
                        value={highRiskCount}
                        icon={<SecurityIcon sx={{ fontSize: 28 }} />}
                        color="#FBBC05"
                        subtitle="Suspicion > 70%"
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={4}>
                    <StatCard
                        title="Cycle Fraud"
                        value={cycleFraudCount}
                        icon={<GroupsIcon sx={{ fontSize: 28 }} />}
                        color="#34A853"
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={4}>
                    <StatCard
                        title="Smurf Fraud"
                        value={smurfFraudCount}
                        icon={<GroupsIcon sx={{ fontSize: 28 }} />}
                        color="#4285F4"
                    />
                </Grid>
                <Grid item xs={12} sm={6} lg={4}>
                    <StatCard
                        title="Shell Fraud"
                        value={shellFraudCount}
                        icon={<GroupsIcon sx={{ fontSize: 28 }} />}
                        color="#FBBC05"
                    />
                </Grid>
            </Grid>

            <Grid container spacing={3}>
                {/* Top Suspicious Accounts */}
                <Grid item xs={12} lg={6}>
                    <Paper elevation={0} sx={{ p: 3, border: '1px solid rgba(255,255,255,0.08)', height: '100%' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                            <TrendingUpIcon sx={{ color: '#EA4335' }} />
                            <Typography variant="h6">Top Suspicious Accounts</Typography>
                        </Box>
                        <Divider sx={{ mb: 2, borderColor: 'rgba(255,255,255,0.08)' }} />
                        {topSuspicious.map(({ account_id, suspicion_score }, index) => (
                            <Box
                                key={account_id}
                                sx={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    py: 1.5,
                                    px: 2,
                                    mb: 1,
                                    borderRadius: 2,
                                    bgcolor: 'rgba(255,255,255,0.03)',
                                    '&:hover': { bgcolor: 'rgba(255,255,255,0.06)' },
                                }}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                    <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 600, minWidth: 24 }}>
                                        #{index + 1}
                                    </Typography>
                                    <Typography variant="body1" fontWeight={500}>{account_id}</Typography>
                                </Box>
                                <Chip
                                    label={`${suspicion_score.toFixed(1)}%`}
                                    size="small"
                                    sx={{
                                        bgcolor: suspicion_score >= 90 ? '#EA433520' : suspicion_score >= 70 ? '#FBBC0520' : '#34A85320',
                                        color: suspicion_score >= 90 ? '#EA4335' : suspicion_score >= 70 ? '#FBBC05' : '#34A853',
                                        fontWeight: 700,
                                    }}
                                />
                            </Box>
                        ))}
                    </Paper>
                </Grid>

                {/* Fraud Rings Overview */}
                <Grid item xs={12} lg={6}>
                    <Paper elevation={0} sx={{ p: 3, border: '1px solid rgba(255,255,255,0.08)', height: '100%' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                            <GroupsIcon sx={{ color: '#4285F4' }} />
                            <Typography variant="h6">Detected Fraud Rings</Typography>
                        </Box>
                        <Divider sx={{ mb: 2, borderColor: 'rgba(255,255,255,0.08)' }} />
                        {fraud_rings.length === 0 ? (
                            <Typography color="text.secondary">No fraud rings detected.</Typography>
                        ) : (
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                                {fraud_rings.slice(0, 5).map((ring) => (
                                    <RingSummaryCard key={ring.ring_id} ring={ring} />
                                ))}
                                {fraud_rings.length > 5 && (
                                    <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mt: 1 }}>
                                        +{fraud_rings.length - 5} more rings. View all in "Fraud Rings" tab.
                                    </Typography>
                                )}
                            </Box>
                        )}
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default HomePage;
