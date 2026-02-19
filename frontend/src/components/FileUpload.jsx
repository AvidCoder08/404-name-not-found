import { useState } from 'react';
import { Box, Typography, Button, Paper, CircularProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { motion } from 'framer-motion';

const FileUpload = ({ onUpload }) => {
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleSubmit = async () => {
        if (!file) return;
        setLoading(true);
        // Simulation of upload for now, or actual call if passed
        if (onUpload) {
            await onUpload(file);
        }
        setLoading(false);
    };

    return (
        <Box
            component={motion.div}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            sx={{ textAlign: 'center', p: 4 }}
        >
            <Typography variant="h4" gutterBottom>
                Analyze Transactions
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
                Upload your transaction CSV file to detect money muling rings and suspicious activities.
            </Typography>

            <Paper
                variant="outlined"
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                sx={{
                    p: 6,
                    mt: 4,
                    border: '2px dashed',
                    borderColor: dragActive ? 'primary.main' : 'divider',
                    backgroundColor: dragActive ? 'action.hover' : 'background.paper',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                }}
                component={motion.div}
                whileHover={{ scale: 1.01 }}
            >
                <input
                    type="file"
                    id="file-upload"
                    style={{ display: 'none' }}
                    onChange={handleChange}
                    accept=".csv"
                />
                <label htmlFor="file-upload">
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                        <CloudUploadIcon sx={{ fontSize: 64, color: 'primary.main' }} />
                        <Typography variant="h6">
                            {file ? file.name : "Drag & Drop or Click to Upload CSV"}
                        </Typography>
                        <Button variant="contained" component="span">
                            Select File
                        </Button>
                    </Box>
                </label>
            </Paper>

            {file && (
                <Button
                    variant="contained"
                    color="primary"
                    size="large"
                    sx={{ mt: 4, px: 6 }}
                    onClick={handleSubmit}
                    disabled={loading}
                >
                    {loading ? <CircularProgress size={24} /> : "Analyze Data"}
                </Button>
            )}
        </Box>
    );
};

export default FileUpload;
