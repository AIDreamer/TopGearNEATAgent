#include <snes/snes.hpp>

#define SMP_CPP
namespace SNES {

SMP smp;

#include "serialization.cpp"
#include "iplrom.cpp"
#include "memory/memory.cpp"
#include "timing/timing.cpp"

void SMP::step(unsigned clocks) {
  clock += clocks * (uint64)cpu.frequency;
  dsp.clock -= clocks;
}

void SMP::synchronize_cpu() {
  if(CPU::Threaded == true) {
    if(clock >= 0 && scheduler.sync != Scheduler::SynchronizeMode::All) co_switch(cpu.thread);
  } else {
    while(clock >= 0) cpu.enter();
  }
}

void SMP::synchronize_cpu_force() {
  if(CPU::Threaded == true) {
    if(clock >= 0 && scheduler.sync != Scheduler::SynchronizeMode::All)
      co_switch(cpu.thread);
    else if(clock >= 0 && scheduler.sync == Scheduler::SynchronizeMode::All)
      interface()->message("SMP had to advance nondeterministically!");
  } else {
    while(clock >= 0) cpu.enter();
  }
}

void SMP::synchronize_dsp() {
  if(DSP::Threaded == true) {
    if(dsp.clock < 0 && scheduler.sync != Scheduler::SynchronizeMode::All) co_switch(dsp.thread);
  } else {
    while(dsp.clock < 0) dsp.enter();
  }
}

void SMP::Enter() { smp.enter(); }

void SMP::enter() {
  while(true) {
    // see comment in timing.cpp
    if(clock > +(768 * 24 * (int64)24000000))
      synchronize_cpu();

    if(scheduler.sync == Scheduler::SynchronizeMode::CPU) {
      synchronize_cpu(); // when in CPU sync mode, always switch back to CPU as soon as possible
    }
    if(scheduler.sync == Scheduler::SynchronizeMode::All) {
      scheduler.exit(Scheduler::ExitReason::SynchronizeEvent);
    }

    debugger.op_exec(regs.pc);
    op_step();
  }
}

void SMP::power() {
  //targets not initialized/changed upon reset
  timer0.target = 0;
  timer1.target = 0;
  timer2.target = 0;

	//zero 01-dec-2012
	//gotta clear these to something, sometime
	dp.w = sp.w = rd.w = wr.w = bit.w = ya.w = 0;
}

void SMP::reset() {
  create(Enter, system.apu_frequency());

  regs.pc = 0xffc0;
  // exact value doesn't matter much, so long as "fetch" is next
  opcode = 0; // NOP
  uindex = 1; // fetch phase

  regs.a = 0x00;
  regs.x = 0x00;
  regs.y = 0x00;
  regs.s = 0xef;
  regs.p = 0x02;

	for(int i=0;i<64*1024;i++) apuram[i] = random(0x00);
  apuram[0x00f4] = 0x00;
  apuram[0x00f5] = 0x00;
  apuram[0x00f6] = 0x00;
  apuram[0x00f7] = 0x00;

  status.clock_counter = 0;
  status.dsp_counter = 0;
  status.timer_step = 3;

  //$00f0
  status.clock_speed = 0;
  status.timer_speed = 0;
  status.timers_enable = true;
  status.ram_disable = false;
  status.ram_writable = true;
  status.timers_disable = false;

  //$00f1
  status.iplrom_enable = true;

  //$00f2
  status.dsp_addr = 0x00;

  //$00f8,$00f9
  status.ram00f8 = 0x00;
  status.ram00f9 = 0x00;

  timer0.stage0_ticks = 0;
  timer1.stage0_ticks = 0;
  timer2.stage0_ticks = 0;

  timer0.stage1_ticks = 0;
  timer1.stage1_ticks = 0;
  timer2.stage1_ticks = 0;

  timer0.stage2_ticks = 0;
  timer1.stage2_ticks = 0;
  timer2.stage2_ticks = 0;

  timer0.stage3_ticks = 0;
  timer1.stage3_ticks = 0;
  timer2.stage3_ticks = 0;

  timer0.current_line = 0;
  timer1.current_line = 0;
  timer2.current_line = 0;

  timer0.enable = false;
  timer1.enable = false;
  timer2.enable = false;
}

SMP::SMP()
	: apuram(nullptr)
{
}

SMP::~SMP() {
	interface()->freeSharedMemory(apuram);
}

void SMP::initialize()
{
	apuram = (uint8*)interface()->allocSharedMemory("APURAM",64 * 1024);
}

}
