import { useEffect, useState } from "react";

export function useMobileLayout(query = "(max-width: 960px)") {
  const [isMobileLayout, setIsMobileLayout] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const mediaQuery = window.matchMedia(query);
    const handleChange = () => setIsMobileLayout(mediaQuery.matches);
    handleChange();
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [query]);

  return isMobileLayout;
}
