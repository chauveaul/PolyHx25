import { Button } from "@/components/ui/button";
import Map from "@/App/Map";

export default function MainApp() {
  return (
    <div className="relative w-[100vw] h-[100vh]">
      <nav className="flex justify-between items-align absolute top-0 left-0 w-[100vw] z-111 text-[#F8F5E9] bg-[#222222] h-[4rem]">
        <p className="ml-[1rem] mt-[0.5rem] text-[2rem]">EmberAlert</p>
        <div className="flex gap-[2rem] mr-[1rem] mt-[0.3rem] items-center">
          <a href="/">Landing Page</a>
          <Button
            variant={"outline"}
            className="bg-[#DE6D13] border-[#DE6D13] hover:bg-[#eba771] hover:border-[#eba771] btn-snapshot"
          >
            Take a snapshot!
          </Button>
        </div>
      </nav>
      <Map />
    </div>
  );
}
