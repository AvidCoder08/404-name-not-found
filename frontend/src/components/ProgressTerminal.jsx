import { useRef, useEffect, useState, useCallback } from 'react';
import { Box, Typography, Paper, LinearProgress, Chip } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import ForceGraph2D from 'react-force-graph-2d';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked';
import LoopIcon from '@mui/icons-material/Loop';

const STEPS = [
    { key: 'upload', label: 'Upload' },
    { key: 'csv', label: 'Parse CSV' },
    { key: 'preprocess', label: 'Preprocess' },
    { key: 'gnn', label: 'GNN Scoring' },
    { key: 'graph', label: 'Build Graph' },
    { key: 'cycles', label: 'Cycles' },
    { key: 'smurfing', label: 'Smurfing' },
    { key: 'shells', label: 'Shells' },
    { key: 'done', label: 'Done' },
];

const StepIndicator = ({ step, currentStep }) => {
    const stepIdx = STEPS.findIndex(s => s.key === step.key);
    const currentIdx = STEPS.findIndex(s => s.key === currentStep);
    const isDone = stepIdx < currentIdx;
    const isActive = stepIdx === currentIdx;

    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {isDone ? (
                <CheckCircleIcon sx={{ fontSize: 16, color: '#B6F2B5' }} />
            ) : isActive ? (
                <LoopIcon
                    sx={{
                        fontSize: 16,
                        color: '#D0BCFF',
                        animation: 'spin 1s linear infinite',
                        '@keyframes spin': {
                            '0%': { transform: 'rotate(0deg)' },
                            '100%': { transform: 'rotate(360deg)' },
                        },
                    }}
                />
            ) : (
                <RadioButtonUncheckedIcon sx={{ fontSize: 16, color: 'rgba(255,255,255,0.2)' }} />
            )}
            <Typography
                variant="caption"
                sx={{
                    color: isDone ? '#B6F2B5' : isActive ? '#D0BCFF' : 'rgba(255,255,255,0.3)',
                    fontWeight: isActive ? 700 : 400,
                    transition: 'all 0.3s ease',
                }}
            >
                {step.label}
            </Typography>
        </Box>
    );
};

const ProgressTerminal = ({ progress, graphData, currentStep, statusMessage }) => {
    const containerRef = useRef(null);
    const fgRef = useRef(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 500 });

    useEffect(() => {
        function handleResize() {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth - 4,
                    height: Math.max(window.innerHeight - 380, 400),
                });
            }
        }
        window.addEventListener('resize', handleResize);
        handleResize();
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // Auto-zoom when graph changes
    useEffect(() => {
        if (fgRef.current && graphData.nodes.length > 0) {
            setTimeout(() => {
                fgRef.current?.zoomToFit(400, 60);
            }, 500);
        }
    }, [graphData.nodes.length]);

    const paintNode = useCallback((node, ctx) => {
        const size = node.val || 3;

        // Glow
        ctx.beginPath();
        ctx.arc(node.x, node.y, size + 3, 0, 2 * Math.PI);
        ctx.fillStyle = (node.color || '#D0BCFF') + '25';
        ctx.fill();

        // Main node
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
        ctx.fillStyle = node.color || '#D0BCFF';
        ctx.fill();

        // Label for large nodes
        if (size > 4) {
            ctx.font = `${Math.max(3, size * 0.7)}px "Google Sans Flex", sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#fff';
            ctx.fillText(node.id, node.x, node.y + size + 5);
        }
    }, []);

    return (
        <Box
            component={motion.div}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            sx={{ width: '100%', maxWidth: '1200px', mx: 'auto', mt: 2 }}
        >
            {/* Progress bar section */}
            <Paper
                elevation={0}
                sx={{
                    p: 3,
                    mb: 2,
                    border: '1px solid rgba(255,255,255,0.08)',
                    background: 'linear-gradient(135deg, rgba(30,27,38,0.95) 0%, rgba(20,18,24,0.95) 100%)',
                }}
            >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
                    <Typography variant="h6" fontWeight={700} sx={{
                        background: 'linear-gradient(135deg, #D0BCFF, #F2B8B5)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                    }}>
                        Analyzing Transactions
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 600 }}>
                        {progress}%
                    </Typography>
                </Box>

                <LinearProgress
                    variant="determinate"
                    value={progress}
                    sx={{
                        height: 6,
                        borderRadius: 3,
                        bgcolor: 'rgba(255,255,255,0.06)',
                        mb: 2,
                        '& .MuiLinearProgress-bar': {
                            borderRadius: 3,
                            background: 'linear-gradient(90deg, #D0BCFF, #F2B8B5)',
                            transition: 'transform 0.4s ease',
                        },
                    }}
                />

                {/* Step indicators */}
                <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
                    {STEPS.map((step) => (
                        <StepIndicator key={step.key} step={step} currentStep={currentStep} />
                    ))}
                </Box>

                {/* Status message */}
                {statusMessage && (
                    <Typography
                        variant="body2"
                        component={motion.p}
                        key={statusMessage}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        sx={{ color: 'text.secondary', mt: 1.5, fontStyle: 'italic' }}
                    >
                        {statusMessage}
                    </Typography>
                )}
            </Paper>

            {/* Live Graph */}
            <Paper
                ref={containerRef}
                elevation={0}
                sx={{
                    overflow: 'hidden',
                    border: '1px solid rgba(255,255,255,0.08)',
                    position: 'relative',
                }}
            >
                <Box sx={{
                    p: 2,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    borderBottom: '1px solid rgba(255,255,255,0.06)',
                }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="subtitle1" fontWeight={600}>
                            Live Network Graph
                        </Typography>
                        {graphData.nodes.length > 0 && (
                            <Chip
                                label={`${graphData.nodes.length} nodes · ${graphData.links.length} links`}
                                size="small"
                                sx={{
                                    bgcolor: 'rgba(208,188,255,0.15)',
                                    color: '#D0BCFF',
                                    fontWeight: 600,
                                    fontSize: '0.7rem',
                                }}
                            />
                        )}
                    </Box>
                    <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#F2B8B5' }} />
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>Critical</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#FFB74D' }} />
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>High</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#D0BCFF' }} />
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>Medium</Typography>
                        </Box>
                    </Box>
                </Box>

                {graphData.nodes.length === 0 ? (
                    <Box sx={{
                        height: dimensions.height,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexDirection: 'column',
                        gap: 2,
                    }}>
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                        >
                            <Box sx={{
                                width: 40, height: 40,
                                borderRadius: '50%',
                                border: '3px solid rgba(208,188,255,0.15)',
                                borderTopColor: '#D0BCFF',
                            }} />
                        </motion.div>
                        <Typography variant="body2" color="text.secondary">
                            Waiting for graph data...
                        </Typography>
                    </Box>
                ) : (
                    <ForceGraph2D
                        ref={fgRef}
                        width={dimensions.width}
                        height={dimensions.height}
                        graphData={graphData}
                        nodeCanvasObject={paintNode}
                        nodePointerAreaPaint={(node, color, ctx) => {
                            ctx.beginPath();
                            ctx.arc(node.x, node.y, (node.val || 3) + 4, 0, 2 * Math.PI);
                            ctx.fillStyle = color;
                            ctx.fill();
                        }}
                        linkColor={() => 'rgba(255,255,255,0.12)'}
                        linkWidth={0.5}
                        linkDirectionalParticles={2}
                        linkDirectionalParticleWidth={1.5}
                        linkDirectionalParticleColor={() => '#D0BCFF'}
                        backgroundColor="transparent"
                        cooldownTicks={60}
                        warmupTicks={20}
                    />
                )}
            </Paper>
        </Box>
    );
};

export default ProgressTerminal;
