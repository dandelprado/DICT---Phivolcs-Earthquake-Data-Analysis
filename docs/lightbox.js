(function(){
  const overlay = document.getElementById('lightbox');
  const leftImg  = document.getElementById('lightbox-left');
  const rightImg = document.getElementById('lightbox-right');
  const caption  = document.getElementById('lightbox-caption');
  const closeBtn = document.getElementById('lightbox-close');

  const PAIRS = {
    "yearly": {
      left:  "ph-yearly-earthquake-counts.png",
      right: "ilocos-yearly-earthquake-counts.png",
      caption: "Earthquakes per Year: Philippines (left) vs Ilocos Norte (right)"
    },
    "monthly": {
      left:  "ph-yearmonth-earthquake-counts-heatmap.png",
      right: "ilocos-monthly-earthquake-counts.png",
      caption: "Earthquakes per Year–Month: Philippines (left) vs Ilocos Norte (right)"
    },
    "mag-dist": {
      left:  "ph-magnitude-distribution.png",
      right: "ilocos-magnitude-distribution.png",
      caption: "Magnitude Distribution: Philippines (left) vs Ilocos Norte (right)"
    },
    "mag-cat": {
      left:  "ph-magnitude-categories.png",
      right: "ilocos-magnitude-categories.png",
      caption: "Magnitude Categories: Philippines (left) vs Ilocos Norte (right)"
    },
    "trend": {
      left:  "ph-magnitude-eventcount-trend.png",
      right: "ilocos-magnitude-eventcount-trend.png",
      caption: "Average Magnitude & Event Count: Philippines (left) vs Ilocos Norte (right)"
    },
    "max-mag": {
      left:  "ph-yearmonth-max-magnitude-heatmap.png",
      right: "ilocos-yearmonth-max-magnitude-heatmap.png",
      caption: "Highest Magnitude per Year–Month: Philippines (left) vs Ilocos Norte (right)"
    }
  };

  function openPair(pairKey){
    const pair = PAIRS[pairKey];
    if(!pair) return;
    leftImg.src  = pair.left;
    rightImg.src = pair.right;
    caption.textContent = pair.caption || "";
    overlay.classList.add('open');
  }

  function closePair(){
    overlay.classList.remove('open');
    leftImg.src = "";
    rightImg.src = "";
    caption.textContent = "";
  }

  document.querySelectorAll('[data-pair]').forEach(img => {
    img.addEventListener('click', () => openPair(img.dataset.pair));
  });

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay || e.target.classList.contains('lightbox-close')) {
      closePair();
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closePair();
  });
})();
