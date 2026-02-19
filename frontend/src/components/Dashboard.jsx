import { Box, Grid, Paper } from '@mui/material';
import { motion } from 'framer-motion';
import ScoreCard from './ScoreCard';
import GraphVisualizer from './GraphVisualizer';
import ResultsTable from './ResultsTable';

const Dashboard = ({ data }) => {
    if (!data) return null;

    return (
        <Box
            component={motion.div}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            sx={{ flexGrow: 1 }}
        >
            <ScoreCard summary={data.summary} />

            <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                    <Grid container spacing={3} direction="column">
                        <Grid item xs={12} style={{ height: '600px' }}>
                            <GraphVisualizer data={data} />
                        </Grid>
                    </Grid>
                </Grid>
                <Grid item xs={12} md={4}>
                    <ResultsTable scores={data.suspicion_scores} />
                </Grid>
            </Grid>
        </Box>
    );
};

export default Dashboard;
