(() => {
  "use strict";
  const config = window.FACA_SITE || {};

  function safeLink(value) {
    if (typeof value !== "string" || !value.trim()) return null;
    try {
      const url = new URL(value, window.location.href);
      return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch (_) { return null; }
  }

  function repositoryFromLocation() {
    const host = window.location.hostname;
    const owner = host.endsWith(".github.io") ? host.slice(0, -10) : "";
    const repository = window.location.pathname.split("/").filter(Boolean)[0];
    if (!owner || !repository || repository.includes(".")) return null;
    return `https://github.com/${owner}/${encodeURIComponent(repository)}`;
  }

  const repositoryUrl = safeLink(config.repositoryUrl) || repositoryFromLocation();
  if (repositoryUrl) {
    document.querySelectorAll("[data-repository-link]").forEach(link => {
      link.href = repositoryUrl;
      link.target = "_blank";
      link.rel = "noopener";
      const label = link.querySelector("[data-repository-label]");
      if (label) label.textContent = "Code";
    });
    document.querySelectorAll("[data-source-path]").forEach(link => {
      link.href = `${repositoryUrl.replace(/\/$/, "")}/blob/${encodeURIComponent(config.sourceBranch || "main")}/${link.dataset.sourcePath}`;
    });
  }
  const paperUrl = safeLink(config.paperUrl);
  if (paperUrl) document.querySelectorAll("[data-paper-link]").forEach(link => { link.href = paperUrl; });

  const gallery = document.querySelector("[data-video-gallery]");
  for (const item of Array.isArray(config.videos) ? config.videos : []) {
    if (!item || typeof item.id !== "string") continue;
    const sourceUrl = safeLink(item.src);
    if (!sourceUrl) continue;
    let slot = Array.from(document.querySelectorAll("[data-video-slot]")).find(el => el.dataset.videoSlot === item.id);
    if (!slot && gallery) {
      const figure = document.createElement("figure");
      slot = document.createElement("div");
      slot.className = "video-stage";
      slot.dataset.videoSlot = item.id;
      const placeholder = document.createElement("div");
      placeholder.className = "video-placeholder";
      const status = document.createElement("p");
      status.className = "placeholder-status";
      status.textContent = "Simulation video";
      placeholder.append(status);
      slot.append(placeholder);
      const caption = document.createElement("figcaption");
      caption.dataset.videoTitle = item.id;
      caption.textContent = item.title || item.id;
      figure.append(slot, caption);
      gallery.append(figure);
    }
    if (!slot) continue;
    const title = Array.from(document.querySelectorAll("[data-video-title]")).find(el => el.dataset.videoTitle === item.id);
    if (title && typeof item.title === "string") title.textContent = item.title;
    const figure = slot.closest("figure");
    if (figure) figure.hidden = false;
    const section = slot.closest(".additional-videos");
    if (section) section.hidden = false;
    const fallback = slot.querySelector(".video-placeholder");
    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "none";
    video.setAttribute("aria-label", item.title || "FACA simulation video");
    const poster = safeLink(item.poster);
    if (poster) video.poster = poster;
    const source = document.createElement("source");
    source.src = sourceUrl;
    source.type = item.type || "video/mp4";
    video.append(source);
    const captions = safeLink(item.captions);
    if (captions) {
      const track = document.createElement("track");
      track.kind = "captions";
      track.label = item.captionsLabel || "English";
      track.srclang = item.captionsLanguage || "en";
      track.src = captions;
      video.append(track);
    }
    const failed = () => {
      video.hidden = true;
      if (fallback) {
        fallback.hidden = false;
        const status = fallback.querySelector(".placeholder-status");
        if (status) status.textContent = "Video currently unavailable";
      }
      slot.setAttribute("aria-label", `${item.title || "Simulation video"} — currently unavailable`);
    };
    video.addEventListener("loadedmetadata", () => {
      if (fallback) fallback.hidden = true;
      slot.setAttribute("aria-label", item.title || "FACA simulation video");
    });
    video.addEventListener("error", failed);
    source.addEventListener("error", failed);
    slot.append(video);
  }
})();
