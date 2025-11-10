import React, { useRef, useEffect, useState } from 'react';

function App() {
  const facialRecognitionModel = process.env.REACT_APP_FACE_RECOGNITION_MODEL || "Facenet";
  const faceDetector = process.env.REACT_APP_DETECTOR_BACKEND || "opencv";
  const distanceMetric = process.env.REACT_APP_DISTANCE_METRIC || "cosine";
  const serviceEndpoint = process.env.REACT_APP_SERVICE_ENDPOINT;
  const antiSpoofing = process.env.REACT_APP_ANTI_SPOOFING === "1";

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [isVerified, setIsVerified] = useState(null);
  const [identity, setIdentity] = useState(null);
  const [accuracy, setAccuracy] = useState(null); // added for match accuracy
  const [isAnalyzed, setIsAnalyzed] = useState(null);
  const [analysis, setAnalysis] = useState([]);
  const [facialDb, setFacialDb] = useState({});

  // Load facial DB
  useEffect(() => {
    const envVarsWithPrefix = {};
    for (const key in process.env) {
      if (key.startsWith("REACT_APP_USER_")) {
        envVarsWithPrefix[key.replace("REACT_APP_USER_", "")] = process.env[key];
      }
    }
    setFacialDb(envVarsWithPrefix);
  }, []);

  // Start webcam once
  useEffect(() => {
    const video = videoRef.current;
    if (video && !video.srcObject) {
      const getVideo = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          video.srcObject = stream;
          await video.play();
        } catch (err) {
          console.error("Error accessing webcam:", err);
        }
      };
      getVideo();
    }
  }, []);

  const captureImage = (task) => {
    setIsVerified(null);
    setIdentity(null);
    setAccuracy(null); // reset accuracy on new capture

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Img = canvas.toDataURL('image/png');

    if (task === "verify") verify(base64Img);
    else if (task === "analyze") analyze(base64Img);
  };

  const verify = async (base64Img) => {
    if (!base64Img) {
      console.error("No image captured for verification.");
      setIsVerified(false);
      setAccuracy(null);
      return;
    }

    let verified = false;
    let foundIdentity = null;
    let matchAccuracy = 0;

    try {
      for (const key in facialDb) {
        const targetEmbedding = facialDb[key];

        if (!targetEmbedding) {
          console.warn(`No reference image/embedding for ${key}, skipping.`);
          continue;
        }

        const requestBody = {
          model_name: facialRecognitionModel,
          detector_backend: faceDetector,
          distance_metric: distanceMetric,
          align: true,
          img1: base64Img,
          img2: targetEmbedding,
          enforce_detection: false,
          anti_spoofing: antiSpoofing,
        };

        console.log("Sending request to verify:", requestBody);

        const response = await fetch(`${serviceEndpoint}/verify`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          console.error("Server error:", response.status, errorData);
          continue;
        }

        const data = await response.json();
        console.log("Verification response:", data);

        // Calculate match accuracy
        const threshold = 0.4; // DeepFace default for cosine distance
        matchAccuracy = Math.max(0, 100 - (data.distance / threshold) * 100);
        matchAccuracy = Math.min(matchAccuracy, 100);

        if (data.verified === true) {
          verified = true;
          foundIdentity = key;
          break;
        }
      }

      setIsVerified(verified);
      setIdentity(foundIdentity);
      setAccuracy(matchAccuracy);

      if (!verified) setIsAnalyzed(null);

    } catch (error) {
      console.error("Exception while verifying image:", error);
      setIsVerified(false);
      setAccuracy(null);
    }
  };

  const analyze = async (base64Img) => {
    setIsAnalyzed(false);
    const result = [];

    try {
      const requestBody = JSON.stringify({
        detector_backend: faceDetector,
        align: true,
        img: base64Img,
        enforce_detection: false,
        anti_spoofing: antiSpoofing,
      });

      const response = await fetch(`${serviceEndpoint}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: requestBody,
      });

      if (!response.ok) {
        console.error("Server error:", response.status);
        return;
      }

      const data = await response.json();

      for (const instance of data.results) {
        const summary = `${instance.age} years old ${instance.dominant_race} ${instance.dominant_gender} with ${instance.dominant_emotion} mood.`;
        result.push(summary);
        console.log(summary);
      }

      if (result.length > 0) {
        setAnalysis(result);
        setIsAnalyzed(true);
        setIsVerified(null);
        setAccuracy(null); // reset accuracy when analyzing
      }

    } catch (error) {
      console.error('Exception while analyzing image:', error);
    }
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '100vh',
      textAlign: 'center',
      backgroundColor: '#282c34',
      color: 'white'
    }}>
      <h1>DeepFace React App</h1>

      {isVerified === true && <p style={{ color: 'green' }}>Verified. Welcome {identity}</p>}
      {isVerified === false && <p style={{ color: 'red' }}>Not Verified</p>}

      {/* Display match accuracy */}
      {isVerified !== null && accuracy !== null && (
        <p>Match Accuracy: {accuracy.toFixed(2)}%</p>
      )}

      {isAnalyzed === true && <p style={{ color: 'green' }}>{analysis.join(', ')}</p>}

      <video ref={videoRef} style={{ width: '100%', maxWidth: '500px' }} />
      <br /><br />
      <button onClick={() => captureImage('verify')}>Verify</button>
      <button onClick={() => captureImage('analyze')}>Analyze</button>
      <br /><br />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}

export default App;
