import { useState } from 'react'
import { AppBar, Toolbar, Typography, Box, IconButton, Alert, Snackbar, Tabs, Tab, Button } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import HubIcon from '@mui/icons-material/Hub';
import TableChartIcon from '@mui/icons-material/TableChart';
import DownloadIcon from '@mui/icons-material/Download';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import SecurityIcon from '@mui/icons-material/Security';
import './App.css'
import FileUpload from './components/FileUpload';
import ProgressTerminal from './components/ProgressTerminal';
import HomePage from './components/HomePage';
import GraphVisualizer from './components/GraphVisualizer';
import FraudRingTable from './components/FraudRingTable';
import DownloadPage from './components/DownloadPage';
import { streamAnalyzeTransactions } from './api';

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [activeTab, setActiveTab] = useState(0);

  const handleUpload = async (file) => {
    setAnalyzing(true);
    setData(null);
    setError(null);
    setProgress(0);
    setLogs(["Initializing upload..."]);

    await streamAnalyzeTransactions(
      file,
      (chunk) => {
        if (chunk.log) setLogs(prev => [...prev, chunk.log]);
        if (chunk.progress) setProgress(chunk.progress);
      },
      (result) => {
        setTimeout(() => {
          setAnalyzing(false);
          setData(result);
          setActiveTab(0);
        }, 1000);
      },
      (errorMessage) => {
        setAnalyzing(false);
        setError(errorMessage);
      }
    );
  };

  const handleReset = () => {
    setData(null);
    setAnalyzing(false);
    setProgress(0);
    setLogs([]);
    setActiveTab(0);
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 0: return <HomePage data={data} />;
      case 1: return <GraphVisualizer data={data} />;
      case 2: return <FraudRingTable data={data} />;
      case 3: return <DownloadPage data={data} />;
      default: return <HomePage data={data} />;
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* App Bar */}
      <AppBar position="sticky" color="transparent" elevation={0}
        sx={{
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          backdropFilter: 'blur(20px)',
          bgcolor: 'rgba(20, 18, 24, 0.85)',
        }}
      >
        <Toolbar sx={{ px: { xs: 2, md: 4 } }}>
          <SecurityIcon sx={{ mr: 1.5, color: 'primary.main' }} />
          <Typography
            variant="h6"
            component="div"
            sx={{
              flexGrow: 1,
              fontWeight: 700,
              background: 'linear-gradient(135deg, #D0BCFF, #F2B8B5)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Money Muling Detector
          </Typography>

          {data && (
            <Button
              variant="outlined"
              size="small"
              startIcon={<RestartAltIcon />}
              onClick={handleReset}
              sx={{
                borderColor: 'rgba(255,255,255,0.15)',
                color: 'text.secondary',
                '&:hover': { borderColor: 'primary.main', color: 'primary.main' },
              }}
            >
              New Analysis
            </Button>
          )}
        </Toolbar>

        {/* Tab Navigation - only show when we have data */}
        {data && (
          <Tabs
            value={activeTab}
            onChange={(e, newValue) => setActiveTab(newValue)}
            sx={{
              px: { xs: 2, md: 4 },
              '& .MuiTabs-indicator': {
                backgroundColor: 'primary.main',
                height: 3,
                borderRadius: '3px 3px 0 0',
              },
            }}
          >
            <Tab icon={<HomeIcon sx={{ fontSize: 18 }} />} iconPosition="start" label="Home" />
            <Tab icon={<HubIcon sx={{ fontSize: 18 }} />} iconPosition="start" label="Network Graph" />
            <Tab icon={<TableChartIcon sx={{ fontSize: 18 }} />} iconPosition="start" label="Fraud Rings" />
            <Tab icon={<DownloadIcon sx={{ fontSize: 18 }} />} iconPosition="start" label="Download" />
          </Tabs>
        )}
      </AppBar>

      {/* Main Content */}
      <Box sx={{ px: { xs: 2, md: 4 }, py: 3, width: '100%' }}>
        {!data && !analyzing ? (
          <FileUpload onUpload={handleUpload} />
        ) : analyzing ? (
          <ProgressTerminal logs={logs} progress={progress} />
        ) : (
          renderTabContent()
        )}
      </Box>

      {/* Error Snackbar */}
      <Snackbar open={!!error} autoHideDuration={6000} onClose={() => setError(null)}>
        <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Box>
  )
}

export default App
