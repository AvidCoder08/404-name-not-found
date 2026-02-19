import { useRef, useEffect } from 'react';
import { Box, Typography, Paper, LinearProgress } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';

const ProgressTerminal = ({ logs, progress }) => {
    const scrollRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <Box
            component={motion.div}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            sx={{ width: '100%', maxWidth: '800px', mx: 'auto', mt: 4 }}
        >
            <Paper
                elevation={4}
                sx={{
                    bgcolor: '#1e1e1e',
                    color: '#00ff41',
                    fontFamily: 'monospace',
                    borderRadius: 2,
                    overflow: 'hidden',
                    border: '1px solid #333'
                }}
            >
                <Box sx={{ bgcolor: '#333', px: 2, py: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#ff5f56' }} />
                    <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#ffbd2e' }} />
                    <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#27c93f' }} />
                    <Typography variant="caption" sx={{ color: '#aaa', ml: 2 }}>
                        analysis_task.exe
                    </Typography>
                </Box>

                <Box sx={{ p: 3 }}>
                    <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" sx={{ color: '#aaa', mb: 0.5 }}>
                            Progress: {progress}%
                        </Typography>
                        <LinearProgress
                            variant="determinate"
                            value={progress}
                            sx={{
                                height: 8,
                                borderRadius: 4,
                                bgcolor: '#333',
                                '& .MuiLinearProgress-bar': {
                                    bgcolor: '#00ff41'
                                }
                            }}
                        />
                    </Box>

                    <Box
                        ref={scrollRef}
                        sx={{
                            height: '300px',
                            overflowY: 'auto',
                            p: 1,
                            border: '1px solid #333',
                            borderRadius: 1,
                            bgcolor: '#000000'
                        }}
                    >
                        {logs.map((log, index) => (
                            <Typography key={index} variant="body2" sx={{ fontFamily: 'monospace', lineHeight: 1.5 }}>
                                <span style={{ color: '#555', marginRight: 8 }}>[{new Date().toLocaleTimeString()}]</span>
                                &gt; {log}
                            </Typography>
                        ))}
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', lineHeight: 1.5 }} className="cursor-blink">
                            <span style={{ color: '#555', marginRight: 8 }}>[{new Date().toLocaleTimeString()}]</span>
                            &gt; <span className="blink">_</span>
                        </Typography>
                    </Box>
                </Box>
            </Paper>
            <style>{`
        @keyframes blink {
          0% { opacity: 0; }
          50% { opacity: 1; }
          100% { opacity: 0; }
        }
        .blink {
          animation: blink 1s infinite;
        }
      `}</style>
        </Box>
    );
};

export default ProgressTerminal;
