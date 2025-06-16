const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const useWebcamBtn = document.getElementById('useWebcamBtn');
const videoUpload = document.getElementById('videoUpload');
const statusBox = document.getElementById('statusBox');

let model = null;
const SCORE_THRESHOLD = 0.9;
let detectionInterval = null;

async function loadModel() {
  console.log('🔄 Loading model...');
  model = await tf.loadGraphModel('model/model.json');
  console.log('✅ Model loaded!');
}

async function useWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  video.play();
  console.log('🎥 Webcam started');
  startLiveDetection();
}

videoUpload.addEventListener('change', () => {
  const file = videoUpload.files[0];
  if (!file) return;

  const fileURL = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = fileURL;
  video.play();
  console.log('📼 Playing uploaded video');

  startLiveDetection();
});

function startLiveDetection() {
  if (detectionInterval) clearInterval(detectionInterval);
  
  detectionInterval = setInterval(() => {
    if (!video.paused && !video.ended) {
      analyzeFrame();
    }
  }, 500); // analyze every 500ms
}

async function analyzeFrame() {
  if (!model) {
    console.warn('⚠️ Model not loaded!');
    return;
  }

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageTensor = tf.browser.fromPixels(canvas).expandDims(0);

  const outputs = await model.executeAsync(imageTensor);

  let scores, classes;
  if (Array.isArray(outputs)) {
    scores = outputs[1].dataSync();
    classes = outputs[2].dataSync().map(c => Math.round(c));
  } else {
    scores = outputs['detection_scores'].dataSync();
    classes = outputs['detection_classes'].dataSync().map(c => Math.round(c));
  }

  tf.dispose([imageTensor, ...Array.isArray(outputs) ? outputs : Object.values(outputs)]);

  const topScore = scores[0];
  const topClass = classes[0];

  console.log('📊 Score:', topScore, 'Class:', topClass);

  if (topScore > SCORE_THRESHOLD) {
    statusBox.innerText = `🚨 Crash detected! (${Math.round(topScore * 100)}%)`;
    statusBox.style.color = 'red';
  } else {
    statusBox.innerText = '✅ No crash detected.';
    statusBox.style.color = 'green';
  }
}

window.onload = async () => {
  await loadModel();
  useWebcamBtn.onclick = useWebcam;
};
