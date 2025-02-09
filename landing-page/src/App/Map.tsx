import React, { useRef, useEffect, useState } from "react";
import { Client, Storage, Functions, ID } from "appwrite";
import * as maptilersdk from "@maptiler/sdk";
import html2canvas from "html2canvas";

export default function Map() {
  const mapContainer = useRef(null);
  const montreal = { lng: -73.613361, lat: 45.504722 };
  const zoom = 14;
  maptilersdk.config.apiKey = import.meta.env.VITE_MAPTILER_API_KEY;

  const client = new Client()
    .setEndpoint("https://cloud.appwrite.io/v1")
    .setProject("67a849a20022902df5d1");

  const storage = new Storage(client);
  const functions = new Functions(client);

  const [render, setRender] = useState(1);
  let counter = 1;

  useEffect(() => {
    const map = new maptilersdk.Map({
      container: mapContainer.current,
      style: maptilersdk.MapStyle.SATELLITE,
      center: [montreal.lng, montreal.lat],
      zoom: zoom,
    });

    map.on("ready", () => {
      console.log("Doc ready");
      document
        .querySelector(".btn-snapshot")
        .addEventListener("click", function () {
          console.log("click");
          takePicture(map);
        });
    });

    async function takePicture(map) {
      map.redraw();
      html2canvas(document.querySelector(".maplibregl-canvas"), {
        useCORS: true,
        allowTaint: false,
      }).then((canvas) => {
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], "map");
            const formData = new FormData();
            formData.append("image", file);
            fetch("http://localhost:5000/", {
              headers: { "Content-Type": "image/jpeg" },
              body: formData,
            }).then((res) => {
              alert(res);
            });
          } else {
            console.log("Blob was null");
          }
        });
      }, "image/jpg");
    }
  }, [montreal.lng, montreal.lat, zoom, mapContainer]);

  return (
    <div
      ref={mapContainer}
      className="absolute w-[100%] h-[100%] map-container"
    />
  );
}
