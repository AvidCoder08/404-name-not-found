import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const analyzeTransactions = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await axios.post(`${API_URL}/analyze`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error("Error analyzing transactions:", error);
        throw error;
    }
};

export const streamAnalyzeTransactions = async (file, onProgress, onComplete, onError) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process buffer line by line
            let lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line

            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const data = JSON.parse(line);

                        if (data.error) {
                            onError(data.log);
                            return;
                        }

                        // Always update progress first
                        onProgress(data);

                        // Then check if this chunk also contains the final result
                        if (data.result) {
                            onComplete(data.result);
                            return; // Stop processing
                        }

                    } catch (e) {
                        console.warn("Skipping invalid JSON chunk:", line);
                    }
                }
            }
        }
    } catch (e) {
        console.error("Stream error found:", e);
        onError(e.message);
    }
};
