# Training Strategies: Staged vs Joint Training

**Visual guide to understanding why staged training produces better models**

---

## 📊 Quick Comparison

<svg width="800" height="300" xmlns="http://www.w3.org/2000/svg">
  <!-- Title -->
  <text x="400" y="30" font-size="20" font-weight="bold" text-anchor="middle">Training Approach Comparison</text>

  <!-- Staged Training -->
  <g>
    <text x="200" y="70" font-size="16" font-weight="bold" text-anchor="middle">Staged Training (Recommended ✅)</text>

    <!-- Stage 1 -->
    <rect x="50" y="90" width="120" height="60" fill="#4CAF50" stroke="#2E7D32" stroke-width="2" rx="5"/>
    <text x="110" y="115" font-size="12" font-weight="bold" text-anchor="middle" fill="white">Stage 1</text>
    <text x="110" y="130" font-size="11" text-anchor="middle" fill="white">Language</text>
    <text x="110" y="145" font-size="10" text-anchor="middle" fill="white">100M tokens</text>

    <!-- Arrow -->
    <path d="M 170 120 L 210 120" stroke="#333" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>

    <!-- Stage 2 -->
    <rect x="210" y="90" width="120" height="60" fill="#2196F3" stroke="#1565C0" stroke-width="2" rx="5"/>
    <text x="270" y="115" font-size="12" font-weight="bold" text-anchor="middle" fill="white">Stage 2</text>
    <text x="270" y="130" font-size="11" text-anchor="middle" fill="white">Code</text>
    <text x="270" y="145" font-size="10" text-anchor="middle" fill="white">7M tokens</text>

    <!-- Results -->
    <rect x="50" y="170" width="280" height="80" fill="#E8F5E9" stroke="#4CAF50" stroke-width="2" rx="5"/>
    <text x="190" y="195" font-size="13" font-weight="bold" text-anchor="middle">Results</text>
    <text x="190" y="215" font-size="12" text-anchor="middle">⏱️ Time: 14-16 hours</text>
    <text x="190" y="235" font-size="12" text-anchor="middle">⭐ Quality: Excellent (loss ~2.0)</text>
  </g>

  <!-- Joint Training -->
  <g>
    <text x="600" y="70" font-size="16" font-weight="bold" text-anchor="middle">Joint Training (Not Recommended ❌)</text>

    <!-- Single Stage -->
    <rect x="450" y="90" width="300" height="60" fill="#FF9800" stroke="#E65100" stroke-width="2" rx="5"/>
    <text x="600" y="115" font-size="12" font-weight="bold" text-anchor="middle">Single Stage</text>
    <text x="600" y="130" font-size="11" text-anchor="middle">Language + Code Mixed</text>
    <text x="600" y="145" font-size="10" text-anchor="middle">107M tokens (93% lang, 7% code)</text>

    <!-- Results -->
    <rect x="450" y="170" width="300" height="80" fill="#FFF3E0" stroke="#FF9800" stroke-width="2" rx="5"/>
    <text x="600" y="195" font-size="13" font-weight="bold" text-anchor="middle">Results</text>
    <text x="600" y="215" font-size="12" text-anchor="middle">⏱️ Time: 12-14 hours</text>
    <text x="600" y="235" font-size="12" text-anchor="middle">⭐ Quality: Poor (loss ~3.5-4.0)</text>
  </g>

  <!-- Arrow marker -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#333"/>
    </marker>
  </defs>
</svg>

---

## 🎓 Curriculum Learning: Why Staged Training Works

<svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
  <!-- Title -->
  <text x="400" y="30" font-size="20" font-weight="bold" text-anchor="middle">Curriculum Learning Principle</text>

  <!-- Human Learning Analogy -->
  <g>
    <text x="200" y="70" font-size="16" font-weight="bold" text-anchor="middle" fill="#1976D2">Human Learning</text>

    <!-- Elementary -->
    <rect x="50" y="90" width="100" height="50" fill="#BBDEFB" stroke="#1976D2" stroke-width="2" rx="5"/>
    <text x="100" y="110" font-size="11" font-weight="bold" text-anchor="middle">Elementary</text>
    <text x="100" y="125" font-size="10" text-anchor="middle">Read & Write</text>

    <!-- Arrow -->
    <path d="M 150 115 L 170 115" stroke="#666" stroke-width="2" marker-end="url(#arrow2)"/>

    <!-- High School -->
    <rect x="170" y="90" width="100" height="50" fill="#90CAF9" stroke="#1976D2" stroke-width="2" rx="5"/>
    <text x="220" y="110" font-size="11" font-weight="bold" text-anchor="middle">High School</text>
    <text x="220" y="125" font-size="10" text-anchor="middle">Subjects</text>

    <!-- Arrow -->
    <path d="M 270 115 L 290 115" stroke="#666" stroke-width="2" marker-end="url(#arrow2)"/>

    <!-- College -->
    <rect x="290" y="90" width="100" height="50" fill="#42A5F5" stroke="#1976D2" stroke-width="2" rx="5"/>
    <text x="340" y="110" font-size="11" font-weight="bold" text-anchor="middle">College</text>
    <text x="340" y="125" font-size="10" text-anchor="middle">Specialize</text>
  </g>

  <!-- Model Learning -->
  <g>
    <text x="200" y="200" font-size="16" font-weight="bold" text-anchor="middle" fill="#388E3C">Model Learning (Staged)</text>

    <!-- Stage 1 -->
    <rect x="50" y="220" width="100" height="50" fill="#C8E6C9" stroke="#388E3C" stroke-width="2" rx="5"/>
    <text x="100" y="240" font-size="11" font-weight="bold" text-anchor="middle">Stage 1</text>
    <text x="100" y="255" font-size="10" text-anchor="middle">Language</text>

    <!-- Arrow -->
    <path d="M 150 245 L 170 245" stroke="#666" stroke-width="2" marker-end="url(#arrow2)"/>

    <!-- Stage 2 -->
    <rect x="170" y="220" width="100" height="50" fill="#81C784" stroke="#388E3C" stroke-width="2" rx="5"/>
    <text x="220" y="240" font-size="11" font-weight="bold" text-anchor="middle">Stage 2</text>
    <text x="220" y="255" font-size="10" text-anchor="middle">Code</text>

    <!-- Arrow -->
    <path d="M 270 245 L 290 245" stroke="#666" stroke-width="2" marker-end="url(#arrow2)"/>

    <!-- Stage 3 -->
    <rect x="290" y="220" width="100" height="50" fill="#4CAF50" stroke="#388E3C" stroke-width="2" rx="5"/>
    <text x="340" y="240" font-size="11" font-weight="bold" text-anchor="middle">Stage 3</text>
    <text x="340" y="255" font-size="10" text-anchor="middle">Tools</text>
  </g>

  <!-- Bad Approach -->
  <g>
    <text x="600" y="200" font-size="16" font-weight="bold" text-anchor="middle" fill="#D32F2F">Joint Training (Bad)</text>

    <rect x="450" y="220" width="300" height="50" fill="#FFCDD2" stroke="#D32F2F" stroke-width="2" rx="5"/>
    <text x="600" y="240" font-size="11" font-weight="bold" text-anchor="middle">Everything at Once</text>
    <text x="600" y="255" font-size="10" text-anchor="middle">Language + Code + Tools → Confused!</text>
  </g>

  <!-- Key Insight -->
  <rect x="50" y="310" width="700" height="70" fill="#FFF9C4" stroke="#F57F17" stroke-width="2" rx="5"/>
  <text x="400" y="335" font-size="14" font-weight="bold" text-anchor="middle">💡 Key Insight</text>
  <text x="400" y="355" font-size="12" text-anchor="middle">Learning step-by-step (curriculum) is more effective than learning everything at once</text>
  <text x="400" y="372" font-size="11" text-anchor="middle" fill="#666">This applies to both humans AND machine learning models!</text>

  <defs>
    <marker id="arrow2" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#666"/>
    </marker>
  </defs>
</svg>

---

## ⚖️ Data Imbalance Problem

<svg width="800" height="450" xmlns="http://www.w3.org/2000/svg">
  <!-- Title -->
  <text x="400" y="30" font-size="20" font-weight="bold" text-anchor="middle">Why Joint Training Fails: Data Imbalance</text>

  <!-- Your Dataset -->
  <text x="150" y="70" font-size="14" font-weight="bold" text-anchor="middle">Your Dataset</text>

  <!-- Language pie slice (93%) -->
  <circle cx="150" cy="150" r="70" fill="#4CAF50" stroke="#2E7D32" stroke-width="2"/>
  <path d="M 150 150 L 150 80 A 70 70 0 1 1 180 85 Z" fill="#2196F3" stroke="#1565C0" stroke-width="2"/>

  <text x="130" y="155" font-size="12" font-weight="bold" fill="white">Language</text>
  <text x="130" y="170" font-size="11" fill="white">100M tokens</text>
  <text x="130" y="185" font-size="13" font-weight="bold" fill="white">93%</text>

  <text x="185" y="100" font-size="10" font-weight="bold" fill="white">Code</text>
  <text x="185" y="112" font-size="9" fill="white">7M</text>
  <text x="185" y="122" font-size="10" fill="white">7%</text>

  <!-- Joint Training Batches -->
  <g>
    <text x="500" y="70" font-size="14" font-weight="bold" text-anchor="middle">Joint Training Batches</text>

    <!-- Batch 1 -->
    <text x="350" y="100" font-size="11">Batch 1:</text>
    <rect x="410" y="88" width="20" height="15" fill="#4CAF50"/>
    <rect x="432" y="88" width="20" height="15" fill="#4CAF50"/>
    <rect x="454" y="88" width="20" height="15" fill="#4CAF50"/>
    <rect x="476" y="88" width="20" height="15" fill="#4CAF50"/>
    <rect x="498" y="88" width="20" height="15" fill="#4CAF50"/>
    <rect x="520" y="88" width="20" height="15" fill="#4CAF50"/>
    <rect x="542" y="88" width="20" height="15" fill="#4CAF50"/>
    <rect x="564" y="88" width="20" height="15" fill="#2196F3"/>

    <!-- Batch 2 -->
    <text x="350" y="125" font-size="11">Batch 2:</text>
    <rect x="410" y="113" width="20" height="15" fill="#4CAF50"/>
    <rect x="432" y="113" width="20" height="15" fill="#4CAF50"/>
    <rect x="454" y="113" width="20" height="15" fill="#4CAF50"/>
    <rect x="476" y="113" width="20" height="15" fill="#4CAF50"/>
    <rect x="498" y="113" width="20" height="15" fill="#4CAF50"/>
    <rect x="520" y="113" width="20" height="15" fill="#4CAF50"/>
    <rect x="542" y="113" width="20" height="15" fill="#4CAF50"/>
    <rect x="564" y="113" width="20" height="15" fill="#4CAF50"/>

    <!-- Batch 3 -->
    <text x="350" y="150" font-size="11">Batch 3:</text>
    <rect x="410" y="138" width="20" height="15" fill="#4CAF50"/>
    <rect x="432" y="138" width="20" height="15" fill="#4CAF50"/>
    <rect x="454" y="138" width="20" height="15" fill="#4CAF50"/>
    <rect x="476" y="138" width="20" height="15" fill="#4CAF50"/>
    <rect x="498" y="138" width="20" height="15" fill="#2196F3"/>
    <rect x="520" y="138" width="20" height="15" fill="#4CAF50"/>
    <rect x="542" y="138" width="20" height="15" fill="#4CAF50"/>
    <rect x="564" y="138" width="20" height="15" fill="#4CAF50"/>

    <text x="350" y="175" font-size="11">...</text>

    <!-- Legend -->
    <rect x="410" y="185" width="20" height="15" fill="#4CAF50" stroke="#333"/>
    <text x="435" y="197" font-size="11">Language example</text>

    <rect x="550" y="185" width="20" height="15" fill="#2196F3" stroke="#333"/>
    <text x="575" y="197" font-size="11">Code example</text>
  </g>

  <!-- Problem -->
  <rect x="50" y="230" width="700" height="80" fill="#FFEBEE" stroke="#C62828" stroke-width="2" rx="5"/>
  <text x="400" y="255" font-size="14" font-weight="bold" text-anchor="middle" fill="#C62828">⚠️ Problem</text>
  <text x="400" y="275" font-size="12" text-anchor="middle">Model sees mostly language examples (93%), barely any code (7%)</text>
  <text x="400" y="293" font-size="12" text-anchor="middle">Result: Learns language well, learns code POORLY</text>

  <!-- Solution -->
  <rect x="50" y="330" width="700" height="100" fill="#E8F5E9" stroke="#2E7D32" stroke-width="2" rx="5"/>
  <text x="400" y="355" font-size="14" font-weight="bold" text-anchor="middle" fill="#2E7D32">✅ Solution: Staged Training</text>
  <text x="400" y="375" font-size="12" text-anchor="middle">Stage 1: 100% language batches → Model masters language</text>
  <text x="400" y="393" font-size="12" text-anchor="middle">Stage 2: 100% code batches → Model focuses entirely on code</text>
  <text x="400" y="411" font-size="12" text-anchor="middle">Result: Excellent at both! 🎉</text>
</svg>

---

## 🔄 Catastrophic Forgetting

<svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
  <!-- Title -->
  <text x="400" y="30" font-size="20" font-weight="bold" text-anchor="middle">Understanding Catastrophic Forgetting</text>

  <!-- Good: Staged Training -->
  <g>
    <text x="200" y="70" font-size="14" font-weight="bold" text-anchor="middle" fill="#388E3C">✅ Staged (Slow, Gentle Learning)</text>

    <!-- Timeline -->
    <line x1="50" y1="150" x2="350" y2="150" stroke="#333" stroke-width="2"/>

    <!-- Stage 1 -->
    <circle cx="100" cy="150" r="30" fill="#4CAF50" stroke="#2E7D32" stroke-width="2"/>
    <text x="100" y="155" font-size="11" font-weight="bold" text-anchor="middle" fill="white">Stage 1</text>
    <text x="100" y="195" font-size="10" text-anchor="middle">Learn</text>
    <text x="100" y="208" font-size="10" text-anchor="middle">Language</text>
    <text x="100" y="225" font-size="9" text-anchor="middle" fill="#666">Loss: 3.4</text>

    <!-- Arrow -->
    <path d="M 130 150 L 170 150" stroke="#666" stroke-width="2" marker-end="url(#arrow3)"/>
    <text x="150" y="140" font-size="9" text-anchor="middle" fill="#666">LR: 5e-6</text>

    <!-- Stage 2 -->
    <circle cx="200" cy="150" r="30" fill="#66BB6A" stroke="#2E7D32" stroke-width="2"/>
    <text x="200" y="150" font-size="10" font-weight="bold" text-anchor="middle" fill="white">Keep</text>
    <text x="200" y="162" font-size="10" font-weight="bold" text-anchor="middle" fill="white">Language</text>
    <text x="200" y="195" font-size="10" text-anchor="middle">+ Add Code</text>
    <text x="200" y="225" font-size="9" text-anchor="middle" fill="#666">Loss: 2.3</text>

    <!-- Arrow -->
    <path d="M 230 150 L 270 150" stroke="#666" stroke-width="2" marker-end="url(#arrow3)"/>

    <!-- Final -->
    <circle cx="300" cy="150" r="30" fill="#81C784" stroke="#2E7D32" stroke-width="2"/>
    <text x="300" y="150" font-size="9" font-weight="bold" text-anchor="middle" fill="white">Language</text>
    <text x="300" y="162" font-size="9" font-weight="bold" text-anchor="middle" fill="white">+ Code</text>
    <text x="300" y="195" font-size="10" text-anchor="middle">Both Good!</text>
    <text x="300" y="225" font-size="9" text-anchor="middle" fill="#2E7D32" font-weight="bold">Loss: 2.0 ✓</text>
  </g>

  <!-- Bad: High Learning Rate -->
  <g>
    <text x="600" y="70" font-size="14" font-weight="bold" text-anchor="middle" fill="#C62828">❌ Too Fast (Catastrophic Forgetting)</text>

    <!-- Timeline -->
    <line x1="450" y1="150" x2="750" y2="150" stroke="#333" stroke-width="2"/>

    <!-- Stage 1 -->
    <circle cx="500" cy="150" r="30" fill="#4CAF50" stroke="#2E7D32" stroke-width="2"/>
    <text x="500" y="155" font-size="11" font-weight="bold" text-anchor="middle" fill="white">Stage 1</text>
    <text x="500" y="195" font-size="10" text-anchor="middle">Learn</text>
    <text x="500" y="208" font-size="10" text-anchor="middle">Language</text>
    <text x="500" y="225" font-size="9" text-anchor="middle" fill="#666">Loss: 3.4</text>

    <!-- Arrow -->
    <path d="M 530 150 L 570 150" stroke="#C62828" stroke-width="3" marker-end="url(#arrow4)"/>
    <text x="550" y="140" font-size="9" text-anchor="middle" fill="#C62828">LR: 1e-5</text>
    <text x="550" y="168" font-size="8" text-anchor="middle" fill="#C62828">TOO FAST!</text>

    <!-- Stage 2 -->
    <circle cx="600" cy="150" r="30" fill="#EF5350" stroke="#C62828" stroke-width="2"/>
    <text x="600" y="150" font-size="9" font-weight="bold" text-anchor="middle" fill="white">Forgot</text>
    <text x="600" y="162" font-size="9" font-weight="bold" text-anchor="middle" fill="white">Language!</text>
    <text x="600" y="195" font-size="10" text-anchor="middle">Trying Code</text>
    <text x="600" y="225" font-size="9" text-anchor="middle" fill="#666">Loss: 6.8</text>

    <!-- Arrow -->
    <path d="M 630 150 L 670 150" stroke="#C62828" stroke-width="2" marker-end="url(#arrow4)"/>

    <!-- Final -->
    <circle cx="700" cy="150" r="30" fill="#F44336" stroke="#C62828" stroke-width="2"/>
    <text x="700" y="155" font-size="9" font-weight="bold" text-anchor="middle" fill="white">Confused!</text>
    <text x="700" y="195" font-size="10" text-anchor="middle">Bad at both</text>
    <text x="700" y="225" font-size="9" text-anchor="middle" fill="#C62828" font-weight="bold">Loss: 6.6 ✗</text>
  </g>

  <!-- Explanation -->
  <rect x="50" y="280" width="700" height="90" fill="#FFF9C4" stroke="#F57F17" stroke-width="2" rx="5"/>
  <text x="400" y="305" font-size="14" font-weight="bold" text-anchor="middle">💡 Why Your Stage 2 Failed</text>
  <text x="400" y="325" font-size="12" text-anchor="middle">Learning rate 1e-5 was too fast → Model forgot language skills while trying to learn code</text>
  <text x="400" y="343" font-size="12" text-anchor="middle">Solution: Use learning rate 5e-6 (half as fast) → Model keeps language, adds code</text>
  <text x="400" y="361" font-size="11" text-anchor="middle" fill="#666">Think of it as gentle fine-tuning, not aggressive retraining</text>

  <defs>
    <marker id="arrow3" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#666"/>
    </marker>
    <marker id="arrow4" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#C62828"/>
    </marker>
  </defs>
</svg>

---

## 📈 Quality vs Time Trade-off

<svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">
  <!-- Title -->
  <text x="400" y="30" font-size="20" font-weight="bold" text-anchor="middle">Quality vs Training Time</text>

  <!-- Axes -->
  <line x1="100" y1="350" x2="700" y2="350" stroke="#333" stroke-width="2"/>
  <line x1="100" y1="80" x2="100" y2="350" stroke="#333" stroke-width="2"/>

  <!-- Y axis label -->
  <text x="40" y="215" font-size="14" font-weight="bold" text-anchor="middle" transform="rotate(-90 40 215)">Model Quality (Lower Loss = Better)</text>

  <!-- X axis label -->
  <text x="400" y="380" font-size="14" font-weight="bold" text-anchor="middle">Training Time (hours)</text>

  <!-- Y axis ticks -->
  <line x1="95" y1="110" x2="105" y2="110" stroke="#333" stroke-width="1"/>
  <text x="85" y="115" font-size="11" text-anchor="end">2.0</text>

  <line x1="95" y1="170" x2="105" y2="170" stroke="#333" stroke-width="1"/>
  <text x="85" y="175" font-size="11" text-anchor="end">3.0</text>

  <line x1="95" y1="230" x2="105" y2="230" stroke="#333" stroke-width="1"/>
  <text x="85" y="235" font-size="11" text-anchor="end">4.0</text>

  <line x1="95" y1="290" x2="105" y2="290" stroke="#333" stroke-width="1"/>
  <text x="85" y="295" font-size="11" text-anchor="end">5.0</text>

  <!-- X axis ticks -->
  <line x1="250" y1="345" x2="250" y2="355" stroke="#333" stroke-width="1"/>
  <text x="250" y="370" font-size="11" text-anchor="middle">10h</text>

  <line x1="400" y1="345" x2="400" y2="355" stroke="#333" stroke-width="1"/>
  <text x="400" y="370" font-size="11" text-anchor="middle">15h</text>

  <line x1="550" y1="345" x2="550" y2="355" stroke="#333" stroke-width="1"/>
  <text x="550" y="370" font-size="11" text-anchor="middle">20h</text>

  <!-- Staged Training Path (green) -->
  <path d="M 100 290 L 250 170 L 400 110" stroke="#4CAF50" stroke-width="4" fill="none" stroke-linecap="round"/>
  <circle cx="100" cy="290" r="6" fill="#4CAF50"/>
  <circle cx="250" cy="170" r="6" fill="#4CAF50"/>
  <circle cx="400" cy="110" r="8" fill="#4CAF50" stroke="#2E7D32" stroke-width="2"/>

  <!-- Labels for staged -->
  <text x="100" y="310" font-size="10" text-anchor="middle" fill="#4CAF50">Start</text>
  <text x="250" y="155" font-size="10" text-anchor="middle" fill="#4CAF50" font-weight="bold">After Stage 1</text>
  <text x="250" y="145" font-size="9" text-anchor="middle" fill="#666">(3.4 loss)</text>
  <text x="400" y="95" font-size="10" text-anchor="middle" fill="#4CAF50" font-weight="bold">After Stage 2</text>
  <text x="400" y="85" font-size="9" text-anchor="middle" fill="#2E7D32">(2.0 loss) ✓</text>

  <!-- Joint Training Path (orange) -->
  <path d="M 100 290 L 370 230" stroke="#FF9800" stroke-width="4" fill="none" stroke-linecap="round" stroke-dasharray="10,5"/>
  <circle cx="370" cy="230" r="8" fill="#FF9800" stroke="#E65100" stroke-width="2"/>

  <!-- Label for joint -->
  <text x="370" y="215" font-size="10" text-anchor="middle" fill="#FF9800" font-weight="bold">Joint Training</text>
  <text x="370" y="205" font-size="9" text-anchor="middle" fill="#666">(3.5 loss) ⚠️</text>

  <!-- Failed Stage 2 (red) -->
  <circle cx="400" cy="260" r="8" fill="#F44336" stroke="#C62828" stroke-width="2"/>
  <text x="400" y="280" font-size="10" text-anchor="middle" fill="#F44336" font-weight="bold">Your Stage 2</text>
  <text x="400" y="290" font-size="9" text-anchor="middle" fill="#C62828">(6.6 loss) ✗</text>

  <!-- Legend -->
  <rect x="520" y="100" width="200" height="110" fill="white" stroke="#666" stroke-width="1" rx="5"/>
  <text x="620" y="120" font-size="12" font-weight="bold" text-anchor="middle">Legend</text>

  <line x1="535" y1="140" x2="565" y2="140" stroke="#4CAF50" stroke-width="3"/>
  <text x="575" y="145" font-size="11">Staged (Recommended)</text>

  <line x1="535" y1="165" x2="565" y2="165" stroke="#FF9800" stroke-width="3" stroke-dasharray="5,3"/>
  <text x="575" y="170" font-size="11">Joint (Poor quality)</text>

  <circle cx="550" cy="190" r="5" fill="#F44336"/>
  <text x="575" y="195" font-size="11">Failed (Wrong LR)</text>
</svg>

---

## 🎯 The Correct Training Path

### Stage 1: Language Pretraining
```bash
python3 scripts/train.py \
  --stage language \
  --architecture dense \
  --model-size large \
  --batch-size 2 \
  --num-epochs 3 \
  --learning-rate 3e-5 \     # Higher LR OK for pretraining
  --warmup-steps 300
```

**Goal:** Learn language fundamentals
**Expected:** Val loss ~3.0-3.5
**Time:** 9-11 hours

---

### Stage 2: Code Fine-tuning
```bash
python3 scripts/train.py \
  --stage code \
  --checkpoint models/language_model_best.pth \
  --batch-size 2 \
  --num-epochs 10 \
  --learning-rate 5e-6 \     # LOWER LR to prevent forgetting!
  --warmup-steps 300
```

**Goal:** Specialize in code while keeping language skills
**Expected:** Val loss ~2.0-2.5
**Time:** 4-5 hours

---

## 📊 Real-World Results

### What Major LLMs Do

| Model | Training Approach |
|-------|------------------|
| **GPT-3/4** | Pretraining (language) → Fine-tuning (instruction) → RLHF |
| **Claude** | Pretraining → Fine-tuning → RLHF → Constitutional AI |
| **Code Llama** | Llama 2 (language) → Code specialization |
| **StarCoder** | Base pretraining → Code fine-tuning |
| **Your model** | Language → Code → Tools → RLHF → ... |

**Common pattern:** Everyone uses staged training!

---

## ✅ Key Takeaways

### ✅ **DO: Use Staged Training**
- Each stage focuses on one objective
- Curriculum learning (easy → hard)
- Better final quality
- Easier to debug
- Industry standard

### ❌ **DON'T: Use Joint Training**
- Model gets confused
- Data imbalance problems
- Worse final quality
- Saves minimal time (~2 hours)
- Not worth the quality loss

### 🎓 **Remember**
- **Learning rate matters!**
  - Stage 1 (pretraining): 3e-5 ✓
  - Stage 2+ (fine-tuning): 5e-6 or lower ✓
  - Too high → Catastrophic forgetting ✗

- **Batch size matters!**
  - Minimum 2 (effective 8 with grad accumulation) ✓
  - Size 1 → Unstable training ✗

- **Patience matters!**
  - More epochs = better convergence ✓
  - Rushing = poor results ✗

---

## 🚀 Your Action Plan

1. **Retrain Stage 2** with corrected settings
2. **Monitor loss:** Should stay close to Stage 1 loss initially
3. **Target:** Val loss ~2.0-2.5 (not 6.6!)
4. **Then:** Proceed to Stage 3 (Tool Calling)

**With proper staged training, you'll build a high-quality model step by step!** 🎉
