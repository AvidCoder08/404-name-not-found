import { useState } from 'react';
import {
    Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
    TablePagination, Typography, Box, Chip, Grid
} from '@mui/material';
import { motion } from 'framer-motion';

const FraudRingTable = ({ data }) => {
    const [page, setPage] = useState(0);
    const [rowsPerPage, setRowsPerPage] = useState(10);

    if (!data || !data.fraud_rings) return null;

    const rings = data.fraud_rings;

    const patternColors = {
        cycle: { bg: '#F2B8B520', color: '#F2B8B5' },
        fan_out: { bg: '#FFB74D20', color: '#FFB74D' },
        fan_in: { bg: '#FFB74D20', color: '#FFB74D' },
        layered_shell: { bg: '#D0BCFF20', color: '#D0BCFF' },
    };

    return (
        <Box
            component={motion.div}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
        >
            <Paper elevation={0} sx={{ width: '100%', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.08)' }}>
                <Box sx={{ p: 3 }}>
                    <Typography variant="h5" fontWeight={600} gutterBottom>
                        Fraud Ring Summary
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        All detected fraud rings with their member accounts, pattern types, and risk scores.
                    </Typography>
                </Box>
                <TableContainer>
                    <Table stickyHeader>
                        <TableHead>
                            <TableRow>
                                <TableCell sx={{ fontWeight: 600, bgcolor: 'background.paper' }}>Ring ID</TableCell>
                                <TableCell sx={{ fontWeight: 600, bgcolor: 'background.paper' }}>Pattern Type</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 600, bgcolor: 'background.paper' }}>Members</TableCell>
                                <TableCell sx={{ fontWeight: 600, bgcolor: 'background.paper' }}>Member Accounts</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 600, bgcolor: 'background.paper' }}>Risk Score</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {rings
                                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                                .map((ring) => {
                                    const pc = patternColors[ring.pattern_type] || { bg: '#CAC4D020', color: '#CAC4D0' };
                                    return (
                                        <TableRow hover key={ring.ring_id} sx={{ '&:hover': { bgcolor: 'rgba(255,255,255,0.03)' } }}>
                                            <TableCell>
                                                <Typography variant="body2" fontWeight={600}>{ring.ring_id}</Typography>
                                            </TableCell>
                                            <TableCell>
                                                <Chip
                                                    label={ring.pattern_type.replace('_', ' ')}
                                                    size="small"
                                                    sx={{ bgcolor: pc.bg, color: pc.color, fontWeight: 600 }}
                                                />
                                            </TableCell>
                                            <TableCell align="center">
                                                <Typography variant="body2">{ring.member_accounts.length}</Typography>
                                            </TableCell>
                                            <TableCell>
                                                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                                                    {ring.member_accounts.slice(0, 4).map(acc => (
                                                        <Chip key={acc} label={acc} size="small" variant="outlined"
                                                            sx={{ borderColor: 'rgba(255,255,255,0.15)', fontSize: '0.75rem' }}
                                                        />
                                                    ))}
                                                    {ring.member_accounts.length > 4 && (
                                                        <Chip
                                                            label={`+${ring.member_accounts.length - 4}`}
                                                            size="small"
                                                            sx={{ bgcolor: 'rgba(255,255,255,0.05)' }}
                                                        />
                                                    )}
                                                </Box>
                                            </TableCell>
                                            <TableCell align="center">
                                                <Chip
                                                    label={`${ring.risk_score}%`}
                                                    size="small"
                                                    sx={{
                                                        bgcolor: ring.risk_score >= 90 ? '#F2B8B520' : '#FFB74D20',
                                                        color: ring.risk_score >= 90 ? '#F2B8B5' : '#FFB74D',
                                                        fontWeight: 700,
                                                    }}
                                                />
                                            </TableCell>
                                        </TableRow>
                                    );
                                })}
                        </TableBody>
                    </Table>
                </TableContainer>
                <TablePagination
                    rowsPerPageOptions={[10, 25, 50]}
                    component="div"
                    count={rings.length}
                    rowsPerPage={rowsPerPage}
                    page={page}
                    onPageChange={(e, p) => setPage(p)}
                    onRowsPerPageChange={(e) => { setRowsPerPage(+e.target.value); setPage(0); }}
                />
            </Paper>
        </Box>
    );
};

export default FraudRingTable;
