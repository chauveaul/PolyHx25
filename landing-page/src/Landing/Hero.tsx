import { Button } from "@/components/ui/button";

export default function Hero() {
  return (
    <div className="bg-[url(/Background.svg)] w-[100vw] h-[100vh] bg-no-repeat font-sans">
      <nav className="flex justify-between items-top p-[1rem] pl-[2rem]">
        <p className="text-[4rem] m-[0rem] text-[#F8F5E9]">EmberAlert</p>
        <div className="flex gap-[2rem] items-center">
          <Button
            variant={"outline"}
            size={"lg"}
            className="bg-[#3A7D44] border-[#3A7D44]"
            onClick={() => (window.location.href = "http://localhost:5173/app")}
          >
            <p className="text-[1.5rem] text-[#F8F5E9]">Get Started</p>
          </Button>
        </div>
      </nav>
      <div className="flex justify-center gap-[12rem] mt-[20vh]">
        <div className="flex flex-col">
          <p className="text-[4rem]">Prevent wildfire like</p>
          <p className="text-[4rem]">a team of 100</p>
          <Button
            variant={"outline"}
            size={"sm"}
            className="bg-[#3A7D44] border-[#3A7D44] max-w-[12rem] p-[2rem]"
            onClick={() => (window.location.href = "http://localhost:5173/app")}
          >
            <p className="text-[#F8F5E9] text-[2rem]">Get Started</p>
          </Button>
        </div>
        <div className="bg-[url(/Trees.svg)] w-[500px] h-[500px] bg-no-repeat"></div>
      </div>
    </div>
  );
}
