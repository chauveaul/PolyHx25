import { Button } from "@/components/ui/button";
import { Client, Storage, Functions, ID } from "appwrite";
import html2canvas from "html2canvas";
import Map from "@/App/Map";

export default function MainApp() {
  const client = new Client()
    .setEndpoint("https://cloud.appwrite.io/v1")
    .setProject("67a849a20022902df5d1");

  const storage = new Storage(client);
  const functions = new Functions(client);

  async function clickHandler() {
    //Creating the file
    const mapEl = document.querySelector(".maplibregl-canvas-container");

    html2canvas(document.querySelector(".maplibregl-canvas"), {
      useCORS: true,
      allowTaint: false,
    }).then((canvas) => {
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], "map");
          const fileId = ID.unique();

          const promise = storage.createFile(
            "67a84b5e002984581076",
            fileId,
            file,
          );

          promise.then(
            (response) => {
              console.log(response); // Success
              const evaluationRes = functions.createExecution(
                "67a84c0400374d85aed7",
                fileId,
              );

              evaluationRes.then(
                (res) => {
                  console.log(res);
                },
                (err) => {
                  console.log(err);
                },
              );
            },
            (error) => {
              console.log(error); // Failure
            },
          );
        } else {
          console.log("Blob was null");
        }
      });
    }, "image/jpg");
  }
  return (
    <div className="relative w-[100vw] h-[100vh]">
      <nav className="flex justify-between items-align absolute top-0 left-0 w-[100vw] z-111 text-[#F8F5E9] bg-[#222222] h-[4rem]">
        <p className="ml-[1rem] mt-[0.5rem] text-[2rem]">EmberAlert</p>
        <div className="flex gap-[2rem] mr-[1rem] mt-[0.3rem] items-center">
          <a>Landing Page</a>
          <Button
            variant={"outline"}
            className="bg-[#DE6D13] border-[#DE6D13] hover:bg-[#eba771] hover:border-[#eba771] btn-snapshot"
            onClick={clickHandler}
          >
            Take a snapshot!
          </Button>
        </div>
      </nav>
      <Map />
    </div>
  );
}
