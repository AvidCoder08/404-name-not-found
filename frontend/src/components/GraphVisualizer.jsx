import React, { useRef, useEffect, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Box, Typography, Paper } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { motion } from 'framer-motion';

const GraphVisualizer = ({ data }) => {
    const fgRef = useRef();
    const theme = useTheme();
    const [graphData, setGraphData] = useState({ nodes: [], links: [] });
    const [dimensions, setDimensions] = useState({ width: 800, height: 700 });
    const containerRef = useRef(null);
    const [highlightNodes, setHighlightNodes] = useState(new Set());
    const [highlightLinks, setHighlightLinks] = useState(new Set());
    const [hoverNode, setHoverNode] = useState(null);

    useEffect(() => {
        if (!data) return;

        const nodes = new Set();
        const links = [];

        if (data.fraud_rings) {
            data.fraud_rings.forEach(ring => {
                const members = ring.member_accounts;
                const type = ring.pattern_type;

                if (type === 'cycle' || type === 'layered_shell') {
                    for (let i = 0; i < members.length; i++) {
                        const source = members[i];
                        const target = members[(i + 1) % members.length];
                        nodes.add(source);
                        nodes.add(target);
                        links.push({ source, target, type });
                    }
                } else if (type === 'fan_out' || type === 'fan_in') {
                    const center = members[members.length - 1];
                    nodes.add(center);
                    for (let i = 0; i < members.length - 1; i++) {
                        const leaf = members[i];
                        nodes.add(leaf);
                        links.push({ source: center, target: leaf, type });
                    }
                } else {
                    for (let i = 0; i < members.length; i++) {
                        const source = members[i];
                        const target = members[(i + 1) % members.length];
                        nodes.add(source);
                        nodes.add(target);
                        links.push({ source, target, type: 'other' });
                    }
                }
            });
        }

        const scoreMap = {};
        if (data.suspicious_accounts) {
            data.suspicious_accounts.forEach(acc => {
                scoreMap[acc.account_id] = acc.suspicion_score;
                if (acc.suspicion_score > 50) nodes.add(acc.account_id);
            });
        }

        const nodesArray = Array.from(nodes).map(id => {
            const score = scoreMap[id] || 0;
            let color;
            if (score >= 90) color = '#F2B8B5';
            else if (score >= 70) color = '#FFB74D';
            else if (score >= 50) color = '#D0BCFF';
            else color = '#CAC4D0';

            return {
                id,
                val: Math.max(score / 10, 2),
                color,
                score,
            };
        });

        setGraphData({ nodes: nodesArray, links });
    }, [data, theme]);

    useEffect(() => {
        function handleResize() {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth - 4,
                    height: Math.max(window.innerHeight - 200, 500),
                });
            }
        }

        window.addEventListener('resize', handleResize);
        handleResize();
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const handleNodeHover = useCallback(node => {
        const newHighlightNodes = new Set();
        const newHighlightLinks = new Set();

        if (node) {
            newHighlightNodes.add(node);
            graphData.links.forEach(link => {
                const s = typeof link.source === 'object' ? link.source : graphData.nodes.find(n => n.id === link.source);
                const t = typeof link.target === 'object' ? link.target : graphData.nodes.find(n => n.id === link.target);
                if (s === node || t === node) {
                    newHighlightLinks.add(link);
                    newHighlightNodes.add(s);
                    newHighlightNodes.add(t);
                }
            });
        }

        setHighlightNodes(newHighlightNodes);
        setHighlightLinks(newHighlightLinks);
        setHoverNode(node || null);
        document.body.style.cursor = node ? 'pointer' : null;
    }, [graphData]);

    const paintNode = useCallback((node, ctx) => {
        const size = node.val || 4;
        const isHighlight = highlightNodes.has(node);

        // Glow effect for highlighted nodes
        if (isHighlight) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, size + 4, 0, 2 * Math.PI);
            ctx.fillStyle = node.color + '40';
            ctx.fill();
        }

        // Main node
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
        ctx.fillStyle = isHighlight ? node.color : node.color + 'AA';
        ctx.fill();

        // Border
        ctx.strokeStyle = isHighlight ? '#fff' : node.color;
        ctx.lineWidth = isHighlight ? 2 : 0.5;
        ctx.stroke();

        // Label
        if (isHighlight || size > 5) {
            ctx.font = `${isHighlight ? 'bold ' : ''}${Math.max(3, size * 0.8)}px "Google Sans Flex", sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#fff';
            ctx.fillText(node.id, node.x, node.y + size + 6);
        }
    }, [highlightNodes]);

    return (
        <Box
            component={motion.div}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            ref={containerRef}
            sx={{ width: '100%' }}
        >
            <Paper
                elevation={0}
                sx={{
                    overflow: 'hidden',
                    border: '1px solid rgba(255,255,255,0.08)',
                    width: '100%',
                }}
            >
                <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="h6" fontWeight={600}>Network Graph</Typography>
                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#F2B8B5' }} />
                            <Typography variant="caption">Critical (&ge;90%)</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#FFB74D' }} />
                            <Typography variant="caption">High (&ge;70%)</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#D0BCFF' }} />
                            <Typography variant="caption">Medium (&ge;50%)</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#CAC4D0' }} />
                            <Typography variant="caption">Low</Typography>
                        </Box>
                    </Box>
                </Box>
                <ForceGraph2D
                    ref={fgRef}
                    width={dimensions.width}
                    height={dimensions.height}
                    graphData={graphData}
                    nodeCanvasObject={paintNode}
                    nodePointerAreaPaint={(node, color, ctx) => {
                        ctx.beginPath();
                        ctx.arc(node.x, node.y, node.val + 4, 0, 2 * Math.PI);
                        ctx.fillStyle = color;
                        ctx.fill();
                    }}
                    linkColor={link => highlightLinks.has(link) ? '#D0BCFF' : 'rgba(255,255,255,0.12)'}
                    linkWidth={link => highlightLinks.has(link) ? 2 : 0.5}
                    linkDirectionalParticles={link => highlightLinks.has(link) ? 4 : 0}
                    linkDirectionalParticleWidth={2}
                    linkDirectionalParticleColor={() => '#D0BCFF'}
                    backgroundColor="transparent"
                    onNodeHover={handleNodeHover}
                    onNodeClick={(node) => {
                        fgRef.current.centerAt(node.x, node.y, 800);
                        fgRef.current.zoom(6, 800);
                    }}
                    cooldownTicks={100}
                    onEngineStop={() => fgRef.current?.zoomToFit(400, 50)}
                />
            </Paper>
        </Box>
    );
};

export default GraphVisualizer;
