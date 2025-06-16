const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const useWebcamBtn = document.getElementById('useWebcamBtn');
const videoUpload = document.getElementById('videoUpload');
const analyzeBtn = document.getElementById('analyzeBtn');

let model = null;
const SCORE_THRESHOLD = 0.9;

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
}

videoUpload.addEventListener('change', () => {
  const file = videoUpload.files[0];
  if (!file) return;

  const fileURL = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = fileURL;
  video.play();
  console.log('📼 Playing uploaded video');
});

async function analyzeFrame() {
  if (!model) {
    alert('⚠️ Model not loaded!');
    return;
  }

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageTensor = tf.browser.fromPixels(canvas).expandDims(0);

  console.log('📦 Running inference...');
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

  console.log('🔍 Top Score:', topScore, 'Class:', topClass);

  if (topScore > SCORE_THRESHOLD) {
    alert(`🚨 Crash detected! Score: ${Math.round(topScore * 100)}%`);
  } else {
    alert('✅ No crash detected.');
  }
}


window.onload = async () => {
  await loadModel();
  useWebcamBtn.onclick = useWebcam;
  analyzeBtn.onclick = analyzeFrame;
};
