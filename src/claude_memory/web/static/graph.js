/**
 * Knowledge Graph Visualization — Pure Canvas force-directed graph
 * No external dependencies. Loaded by index.html.
 */

(function () {
  "use strict";

  // ── Color maps ──────────────────────────────────────────────────────────────
  const NODE_COLORS = {
    decision:   "#00d4ff",
    todo:       "#ffd700",
    pattern:    "#00ff88",
    issue:      "#ff4444",
    solution:   "#4488ff",
    preference: "#ff44ff",
    context:    "#888888",
    learning:   "#ffffff",
  };

  const EDGE_COLORS = {
    shared_tag:    "#444444",
    same_session:  "#666666",
    similar_title: "#0f3460",
    cross_project: "#e94560",
  };

  // ── Physics constants ───────────────────────────────────────────────────────
  const REPULSION_K     = 6000;   // Coulomb constant
  const SPRING_K        = 0.005;  // Hooke spring stiffness
  const REST_LENGTH     = 120;    // Ideal edge length
  const DAMPING         = 0.85;   // Velocity decay per tick
  const CENTER_GRAVITY  = 0.01;   // Pull toward canvas center
  const MAX_VELOCITY    = 8;      // Clamp speed to avoid explosions
  const MIN_ALPHA       = 0.001;  // Simulation "cool-down" threshold

  // ── Module state ────────────────────────────────────────────────────────────
  let canvas, ctx;
  let nodes = [], edges = [];
  let width = 0, height = 0;

  // Camera / transform
  let scale   = 1;
  let offsetX = 0, offsetY = 0;

  // Interaction state
  let dragNode     = null;
  let isPanning    = false;
  let panStartX    = 0, panStartY = 0;
  let hoverNode    = null;
  let selectedNode = null;

  // Simulation bookkeeping
  let alpha   = 1;   // "temperature" — decays toward 0
  let running = false;

  // Filters
  let visibleNodeTypes = new Set(Object.keys(NODE_COLORS));
  let visibleEdgeTypes = new Set(Object.keys(EDGE_COLORS));

  // ── Helpers ─────────────────────────────────────────────────────────────────
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function truncate(str, max) {
    if (!str) return "";
    return str.length > max ? str.slice(0, max - 1) + "\u2026" : str;
  }

  function nodeRadius(node) {
    const imp = node.importance != null ? node.importance : 0.5;
    return 8 + imp * 17;  // 8..25
  }

  function edgeWidth(edge) {
    const w = edge.weight != null ? edge.weight : 0.5;
    return 1 + w * 2;     // 1..3
  }

  function nodeColor(node) {
    return NODE_COLORS[node.type] || "#aaaaaa";
  }

  function edgeColor(edge) {
    return EDGE_COLORS[edge.relationship] || "#555555";
  }

  // Convert page coords → graph world coords
  function toWorld(px, py) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (px - rect.left - offsetX) / scale,
      y: (py - rect.top  - offsetY) / scale,
    };
  }

  function nodeAt(wx, wy) {
    // Iterate in reverse so top-drawn nodes are picked first
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i];
      if (!visibleNodeTypes.has(n.type)) continue;
      const r  = nodeRadius(n);
      const dx = n.x - wx;
      const dy = n.y - wy;
      if (dx * dx + dy * dy <= r * r) return n;
    }
    return null;
  }

  // ── Force simulation ───────────────────────────────────────────────────────
  function applyForces() {
    const len = nodes.length;

    // 1. Repulsion between every pair of visible nodes
    for (let i = 0; i < len; i++) {
      const a = nodes[i];
      if (!visibleNodeTypes.has(a.type)) continue;
      for (let j = i + 1; j < len; j++) {
        const b = nodes[j];
        if (!visibleNodeTypes.has(b.type)) continue;

        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;

        const force = REPULSION_K / (dist * dist);
        const fx = (dx / dist) * force * alpha;
        const fy = (dy / dist) * force * alpha;

        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }

    // 2. Spring attraction along edges
    for (const edge of edges) {
      if (!visibleEdgeTypes.has(edge.relationship)) continue;
      const a = edge.sourceNode;
      const b = edge.targetNode;
      if (!a || !b) continue;
      if (!visibleNodeTypes.has(a.type) || !visibleNodeTypes.has(b.type)) continue;

      let dx = b.x - a.x;
      let dy = b.y - a.y;
      let dist = Math.sqrt(dx * dx + dy * dy) || 1;

      const displacement = dist - REST_LENGTH;
      const force = SPRING_K * displacement * alpha;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;

      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }

    // 3. Center gravity
    const cx = width  / 2 / scale - offsetX / scale;
    const cy = height / 2 / scale - offsetY / scale;

    for (const n of nodes) {
      if (!visibleNodeTypes.has(n.type)) continue;
      n.vx += (cx - n.x) * CENTER_GRAVITY * alpha;
      n.vy += (cy - n.y) * CENTER_GRAVITY * alpha;
    }

    // 4. Integrate velocity → position; apply damping
    for (const n of nodes) {
      if (n === dragNode) continue;  // dragged node is pinned
      n.vx *= DAMPING;
      n.vy *= DAMPING;
      // Clamp velocity
      const speed = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
      if (speed > MAX_VELOCITY) {
        n.vx = (n.vx / speed) * MAX_VELOCITY;
        n.vy = (n.vy / speed) * MAX_VELOCITY;
      }
      n.x += n.vx;
      n.y += n.vy;
    }

    // Cool down
    alpha *= 0.995;
    if (alpha < MIN_ALPHA) alpha = MIN_ALPHA;
  }

  // ── Rendering ──────────────────────────────────────────────────────────────
  function render() {
    ctx.save();
    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = "#0a0a1a";
    ctx.fillRect(0, 0, width, height);

    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);

    drawEdges();
    drawNodes();

    ctx.restore();

    drawTooltip();
    drawHUD();
  }

  function drawEdges() {
    // Pre-compute edge counts between same node pairs for curving
    const pairCounts = {};
    const pairIndex  = {};
    for (const e of edges) {
      if (!visibleEdgeTypes.has(e.relationship)) continue;
      const a = e.sourceNode, b = e.targetNode;
      if (!a || !b) continue;
      if (!visibleNodeTypes.has(a.type) || !visibleNodeTypes.has(b.type)) continue;

      const key = a.id < b.id ? a.id + "|" + b.id : b.id + "|" + a.id;
      pairCounts[key] = (pairCounts[key] || 0) + 1;
      pairIndex[key]  = 0;
    }

    for (const e of edges) {
      if (!visibleEdgeTypes.has(e.relationship)) continue;
      const a = e.sourceNode, b = e.targetNode;
      if (!a || !b) continue;
      if (!visibleNodeTypes.has(a.type) || !visibleNodeTypes.has(b.type)) continue;

      const key   = a.id < b.id ? a.id + "|" + b.id : b.id + "|" + a.id;
      const total = pairCounts[key];
      const idx   = pairIndex[key]++;

      const isHighlighted = selectedNode &&
        (a === selectedNode || b === selectedNode);

      ctx.beginPath();
      ctx.strokeStyle = isHighlighted ? "#ffffff" : edgeColor(e);
      ctx.lineWidth   = edgeWidth(e) / scale;   // keep apparent width stable
      ctx.globalAlpha = isHighlighted ? 1 : (selectedNode ? 0.15 : 0.6);

      if (total > 1) {
        // Curved edge — offset perpendicular to the line
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const nx = -dy / len;
        const ny =  dx / len;
        const curveOffset = (idx - (total - 1) / 2) * 30;

        const cpx = mx + nx * curveOffset;
        const cpy = my + ny * curveOffset;

        ctx.moveTo(a.x, a.y);
        ctx.quadraticCurveTo(cpx, cpy, b.x, b.y);
      } else {
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function drawNodes() {
    for (const n of nodes) {
      if (!visibleNodeTypes.has(n.type)) continue;

      const r  = nodeRadius(n);
      const isSelected   = n === selectedNode;
      const isNeighbor   = selectedNode && n._isNeighbor;
      const isDimmed     = selectedNode && !isSelected && !isNeighbor;
      const isHovered    = n === hoverNode;

      // Glow for selected / hovered
      if (isSelected || isHovered) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 6, 0, Math.PI * 2);
        ctx.fillStyle = nodeColor(n);
        ctx.globalAlpha = 0.25;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      // Circle body
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fillStyle   = nodeColor(n);
      ctx.globalAlpha = isDimmed ? 0.15 : 1;
      ctx.fill();

      // Border
      ctx.lineWidth   = isSelected ? 3 / scale : 1.5 / scale;
      ctx.strokeStyle = isSelected ? "#ffffff" : "rgba(0,0,0,0.5)";
      ctx.stroke();

      // Label
      const fontSize = clamp(11 / scale, 8, 16);
      ctx.font      = `${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
      ctx.textAlign  = "center";
      ctx.fillStyle  = isDimmed ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.9)";
      ctx.fillText(truncate(n.title || n.id, 20), n.x, n.y + r + fontSize + 2);

      ctx.globalAlpha = 1;
    }
  }

  function drawTooltip() {
    if (!hoverNode) return;

    const n    = hoverNode;
    const rect = canvas.getBoundingClientRect();
    const sx   = n.x * scale + offsetX;
    const sy   = n.y * scale + offsetY;

    const lines = [
      n.title || n.id,
      `Type: ${n.type || "unknown"}`,
      n.project ? `Project: ${n.project}` : null,
      n.importance != null ? `Importance: ${(n.importance * 100).toFixed(0)}%` : null,
    ].filter(Boolean);

    ctx.save();
    const pad  = 8;
    const lineH = 16;
    const font = "13px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.font   = font;

    const maxW = Math.max(...lines.map(l => ctx.measureText(l).width));
    const boxW = maxW + pad * 2;
    const boxH = lines.length * lineH + pad * 2;

    let tx = sx + 14;
    let ty = sy - boxH / 2;
    // Keep within canvas
    if (tx + boxW > width)  tx = sx - boxW - 14;
    if (ty < 4)             ty = 4;
    if (ty + boxH > height) ty = height - boxH - 4;

    // Background
    ctx.fillStyle = "rgba(10, 10, 30, 0.92)";
    roundRect(ctx, tx, ty, boxW, boxH, 6);
    ctx.fill();

    ctx.strokeStyle = nodeColor(n);
    ctx.lineWidth   = 1.5;
    roundRect(ctx, tx, ty, boxW, boxH, 6);
    ctx.stroke();

    // Text
    ctx.fillStyle = "#e0e0e0";
    ctx.textAlign = "left";
    ctx.font      = font;
    lines.forEach((line, i) => {
      if (i === 0) ctx.fillStyle = "#ffffff";
      else         ctx.fillStyle = "#b0b0b0";
      ctx.fillText(line, tx + pad, ty + pad + (i + 1) * lineH - 3);
    });
    ctx.restore();
  }

  function drawHUD() {
    const visibleNodes = nodes.filter(n => visibleNodeTypes.has(n.type)).length;
    const visibleEdges = edges.filter(e => {
      if (!visibleEdgeTypes.has(e.relationship)) return false;
      const a = e.sourceNode, b = e.targetNode;
      return a && b && visibleNodeTypes.has(a.type) && visibleNodeTypes.has(b.type);
    }).length;

    ctx.save();
    ctx.font      = "12px -apple-system, BlinkMacSystemFont, monospace";
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.textAlign = "left";
    ctx.fillText(`Nodes: ${visibleNodes}  Edges: ${visibleEdges}  Zoom: ${(scale * 100).toFixed(0)}%`, 10, height - 10);
    ctx.restore();
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y,     x + w, y + r,     r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h,     x, y + h - r,     r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y,         x + r, y,         r);
    ctx.closePath();
  }

  // ── Main loop ──────────────────────────────────────────────────────────────
  function tick() {
    if (!running) return;
    applyForces();
    render();
    requestAnimationFrame(tick);
  }

  // ── Data initialization ────────────────────────────────────────────────────
  function initSimulation(rawNodes, rawEdges) {
    // Build node map
    const nodeMap = {};
    nodes = rawNodes.map(n => {
      const node = {
        ...n,
        x:  n.x != null ? n.x : (Math.random() - 0.5) * width  * 0.6 + width  / 2,
        y:  n.y != null ? n.y : (Math.random() - 0.5) * height * 0.6 + height / 2,
        vx: 0,
        vy: 0,
        _isNeighbor: false,
      };
      nodeMap[node.id] = node;
      return node;
    });

    edges = rawEdges.map(e => ({
      ...e,
      sourceNode: nodeMap[e.source],
      targetNode: nodeMap[e.target],
    })).filter(e => e.sourceNode && e.targetNode);

    alpha   = 1;
    running = true;
    selectedNode = null;
    hoverNode    = null;
    requestAnimationFrame(tick);
  }

  function markNeighbors() {
    // Reset
    for (const n of nodes) n._isNeighbor = false;

    if (!selectedNode) return;

    for (const e of edges) {
      if (!visibleEdgeTypes.has(e.relationship)) continue;
      if (e.sourceNode === selectedNode) e.targetNode._isNeighbor = true;
      if (e.targetNode === selectedNode) e.sourceNode._isNeighbor = true;
    }
    selectedNode._isNeighbor = true;
  }

  // ── Event handlers ─────────────────────────────────────────────────────────
  function onMouseDown(e) {
    const w = toWorld(e.clientX, e.clientY);
    const hit = nodeAt(w.x, w.y);

    if (hit) {
      dragNode = hit;
      dragNode.vx = 0;
      dragNode.vy = 0;
      canvas.style.cursor = "grabbing";
    } else {
      isPanning = true;
      panStartX = e.clientX - offsetX;
      panStartY = e.clientY - offsetY;
      canvas.style.cursor = "move";
    }
  }

  function onMouseMove(e) {
    const w = toWorld(e.clientX, e.clientY);

    if (dragNode) {
      dragNode.x = w.x;
      dragNode.y = w.y;
      alpha = Math.max(alpha, 0.05);  // keep simulation warm while dragging
      return;
    }

    if (isPanning) {
      offsetX = e.clientX - panStartX;
      offsetY = e.clientY - panStartY;
      render();
      return;
    }

    // Hover detection
    const hit = nodeAt(w.x, w.y);
    if (hit !== hoverNode) {
      hoverNode = hit;
      canvas.style.cursor = hit ? "pointer" : "default";
      if (!running) render();  // still render tooltip when sim is cold
    }
  }

  function onMouseUp(e) {
    if (dragNode) {
      canvas.style.cursor = "pointer";
      dragNode = null;
    } else if (isPanning) {
      isPanning = false;
      canvas.style.cursor = "default";
    }
  }

  function onClick(e) {
    if (dragNode || isPanning) return;

    const w   = toWorld(e.clientX, e.clientY);
    const hit = nodeAt(w.x, w.y);

    if (hit) {
      selectedNode = (selectedNode === hit) ? null : hit;
    } else {
      selectedNode = null;
    }
    markNeighbors();
    if (!running) render();
  }

  function onWheel(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
    const newScale   = clamp(scale * zoomFactor, 0.1, 5);
    const ratio      = newScale / scale;

    // Zoom toward cursor
    offsetX = mx - (mx - offsetX) * ratio;
    offsetY = my - (my - offsetY) * ratio;
    scale   = newScale;

    if (!running) render();
  }

  function resize() {
    const parent = canvas.parentElement;
    width  = parent.clientWidth;
    height = parent.clientHeight;
    canvas.width  = width  * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    canvas.style.width  = width  + "px";
    canvas.style.height = height + "px";
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    if (!running) render();
  }

  // ── Controls panel ─────────────────────────────────────────────────────────
  function buildControls(container) {
    const panel = document.createElement("div");
    panel.id = "graph-controls";
    panel.style.cssText = `
      position: absolute; top: 10px; right: 10px; z-index: 10;
      background: rgba(10,10,30,0.88); border: 1px solid #333;
      border-radius: 8px; padding: 12px 14px; color: #ccc;
      font: 12px -apple-system, BlinkMacSystemFont, sans-serif;
      max-height: 90%; overflow-y: auto; min-width: 170px;
    `;

    // ── Node type filters
    panel.appendChild(heading("Node Types"));
    for (const type of Object.keys(NODE_COLORS)) {
      panel.appendChild(filterCheckbox(type, NODE_COLORS[type], visibleNodeTypes));
    }

    // ── Edge type filters
    panel.appendChild(heading("Edge Types"));
    for (const type of Object.keys(EDGE_COLORS)) {
      panel.appendChild(filterCheckbox(type, EDGE_COLORS[type], visibleEdgeTypes));
    }

    // ── Reset layout button
    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset Layout";
    resetBtn.style.cssText = `
      display: block; width: 100%; margin-top: 12px; padding: 6px 0;
      background: #1a1a3a; color: #ccc; border: 1px solid #444;
      border-radius: 4px; cursor: pointer; font-size: 12px;
    `;
    resetBtn.addEventListener("click", () => {
      scale = 1; offsetX = 0; offsetY = 0;
      selectedNode = null; hoverNode = null;
      for (const n of nodes) {
        n.x  = (Math.random() - 0.5) * width * 0.6 + width / 2;
        n.y  = (Math.random() - 0.5) * height * 0.6 + height / 2;
        n.vx = 0; n.vy = 0;
      }
      alpha = 1;
      markNeighbors();
      if (!running) { running = true; requestAnimationFrame(tick); }
    });
    panel.appendChild(resetBtn);

    container.appendChild(panel);
  }

  function heading(text) {
    const h = document.createElement("div");
    h.textContent = text;
    h.style.cssText = "font-weight:600;margin:8px 0 4px;color:#fff;font-size:12px;";
    return h;
  }

  function filterCheckbox(label, color, set) {
    const wrap = document.createElement("label");
    wrap.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0;cursor:pointer;";

    const cb = document.createElement("input");
    cb.type    = "checkbox";
    cb.checked = true;
    cb.style.accentColor = color;
    cb.addEventListener("change", () => {
      if (cb.checked) set.add(label); else set.delete(label);
      markNeighbors();
      alpha = Math.max(alpha, 0.05);
      if (!running) { running = true; requestAnimationFrame(tick); }
    });

    const swatch = document.createElement("span");
    swatch.style.cssText = `display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};`;

    const txt = document.createElement("span");
    txt.textContent = label;
    txt.style.cssText = "font-size:11px;";

    wrap.appendChild(cb);
    wrap.appendChild(swatch);
    wrap.appendChild(txt);
    return wrap;
  }

  // ── API integration ────────────────────────────────────────────────────────
  async function loadGraph(projectFilter) {
    const url = projectFilter
      ? "/api/graph?project=" + encodeURIComponent(projectFilter)
      : "/api/graph";

    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const data = await resp.json();
      // data = { nodes: [...], edges: [...], summary: {...} }
      initSimulation(data.nodes || [], data.edges || []);
    } catch (err) {
      console.warn("[graph] Failed to load graph data:", err);
      // Show friendly empty state
      running = false;
      render();
      ctx.save();
      ctx.font      = "16px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillStyle = "#666";
      ctx.textAlign = "center";
      ctx.fillText("No graph data available", width / 2, height / 2);
      ctx.fillStyle = "#444";
      ctx.font      = "13px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillText(err.message, width / 2, height / 2 + 24);
      ctx.restore();
    }
  }

  // ── Public entry point ─────────────────────────────────────────────────────
  /**
   * Called by index.html when the Graph tab is selected.
   * @param {string} containerId  - DOM id of the wrapper element
   * @param {string|null} projectFilter - optional project name to scope the graph
   */
  window.initGraphVisualization = function (containerId, projectFilter) {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error("[graph] Container not found:", containerId);
      return;
    }

    // Ensure container has positioning context for overlay controls
    if (getComputedStyle(container).position === "static") {
      container.style.position = "relative";
    }

    // Clean up any previous instance
    const oldCanvas   = container.querySelector("canvas");
    const oldControls = container.querySelector("#graph-controls");
    if (oldCanvas)   oldCanvas.remove();
    if (oldControls) oldControls.remove();
    running = false;

    // Create canvas
    canvas = document.createElement("canvas");
    canvas.id = "graph-canvas";
    canvas.style.cssText = "display:block;width:100%;height:100%;";
    container.appendChild(canvas);

    ctx = canvas.getContext("2d");

    // Size to container
    width  = container.clientWidth  || 800;
    height = container.clientHeight || 600;
    canvas.width  = width  * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    canvas.style.width  = width  + "px";
    canvas.style.height = height + "px";
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

    // Reset camera
    scale = 1; offsetX = 0; offsetY = 0;

    // Bind events
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseup",   onMouseUp);
    canvas.addEventListener("click",     onClick);
    canvas.addEventListener("wheel",     onWheel, { passive: false });
    window.addEventListener("resize",    resize);

    // Build filter controls
    visibleNodeTypes = new Set(Object.keys(NODE_COLORS));
    visibleEdgeTypes = new Set(Object.keys(EDGE_COLORS));
    buildControls(container);

    // Initial render (blank background)
    render();

    // Fetch graph data
    loadGraph(projectFilter || null);
  };
})();
