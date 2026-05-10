import { icons as carbonIcons } from "@iconify-json/carbon";

function pick(name: keyof typeof carbonIcons.icons) {
  return {
    body: carbonIcons.icons[name].body,
    width: carbonIcons.width,
    height: carbonIcons.height,
  };
}

export const reviewIcon = pick("review");
export const imageIcon = pick("image");
export const checkIcon = pick("checkmark-outline");
export const closeIcon = pick("close-outline");
export const dropIcon = pick("close");
export const nextIcon = pick("next-outline");
export const positiveIcon = pick("checkmark-filled");
export const negativeIcon = pick("close-filled");
export const connectIcon = pick("connect");
export const playIcon = pick("play-outline");
export const downloadIcon = pick("download");
export const terminalIcon = pick("terminal");
export const serverIcon = pick("server-proxy");
export const warningIcon = pick("warning-alt");
export const mapIcon = pick("map");
export const satelliteIcon = pick("satellite");
