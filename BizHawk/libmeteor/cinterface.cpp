#include "ameteor.hpp"
#include "ameteor/cartmem.hpp"
#include "source/debug.hpp"
#include <sstream>

#define EXPORT extern "C" __declspec(dllexport)

void (*messagecallback)(const char *msg, int abort) = NULL;

EXPORT void libmeteor_setmessagecallback(void (*callback)(const char *msg, int abort))
{
	messagecallback = callback;
	print_bizhawk("libmeteor message stream operational.");
}

void print_bizhawk(const char *msg)
{
	if (messagecallback)
		messagecallback(msg, 0);
}
void print_bizhawk(std::string &msg)
{
	if (messagecallback)
		messagecallback(msg.c_str(), 0);
}
void abort_bizhawk(const char *msg)
{
	if (messagecallback)
		messagecallback(msg, 1);
	AMeteor::Stop(); // makes it easy to pick apart what happened
}

uint16_t (*keycallback)() = NULL;

void keyupdate_bizhawk()
{
	if (keycallback)
		AMeteor::_keypad.SetPadState(keycallback() ^ 0x3FF);
}

EXPORT void libmeteor_setkeycallback(uint16_t (*callback)())
{
	keycallback = callback;
}

bool traceenabled = false;
void (*tracecallback)(const char *msg) = NULL;

EXPORT void libmeteor_settracecallback(void (*callback)(const char*msg))
{
	tracecallback = callback;
	traceenabled = tracecallback != NULL;
}

void trace_bizhawk(std::string msg)
{
	if (tracecallback)
		tracecallback(msg.c_str());
}

EXPORT void libmeteor_hardreset()
{
	AMeteor::Reset(AMeteor::UNIT_ALL ^ (AMeteor::UNIT_MEMORY_BIOS | AMeteor::UNIT_MEMORY_ROM));
}

uint32_t *videobuff;

void videocb(const uint16_t *frame)
{
	uint32_t *dest = videobuff;
	const uint16_t *src = frame;
	for (int i = 0; i < 240 * 160; i++, src++, dest++)
	{
		uint16_t c = *src;
		uint16_t b = c >> 10 & 31;
		uint16_t g = c >> 5 & 31;
		uint16_t r = c & 31;
		b = b << 3 | b >> 2;
		g = g << 3 | g >> 2;
		r = r << 3 | r >> 2;
		*dest = b | g << 8 | r << 16 | 0xff000000;
	}
	AMeteor::Stop(); // to the end of frame only
}

int16_t *soundbuff;
int16_t *soundbuffcur;
int16_t *soundbuffend;

void soundcb(const int16_t *samples)
{
	if (soundbuffcur < soundbuffend)
	{
		*soundbuffcur++ = *samples++;
		*soundbuffcur++ = *samples++;
	}
}

EXPORT unsigned libmeteor_emptysound()
{
	unsigned ret = (soundbuffcur - soundbuff) * sizeof(int16_t);
	soundbuffcur = soundbuff;
	return ret;
}

EXPORT int libmeteor_setbuffers(uint32_t *vid, unsigned vidlen, int16_t *aud, unsigned audlen)
{
	if (vidlen < 240 * 160 * sizeof(uint32_t))
		return 0;
	if (audlen < 4 || audlen % 4 != 0)
		return 0;
	videobuff = vid;
	soundbuff = aud;
	soundbuffend = soundbuff + audlen / sizeof(int16_t);
	libmeteor_emptysound();
	return 1;
}

EXPORT void libmeteor_init()
{
	static bool first = true;
	if (first)
	{
		AMeteor::_lcd.GetScreen().GetRenderer().SetFrameSlot(syg::ptr_fun(videocb));
		AMeteor::_sound.GetSpeaker().SetFrameSlot(syg::ptr_fun(soundcb));
		first = false;
	}
}

EXPORT void libmeteor_frameadvance()
{
	AMeteor::Run(10000000);
}

EXPORT void libmeteor_loadrom(const void *data, unsigned size)
{
	AMeteor::_memory.LoadRom((const uint8_t*)data, size);
}

EXPORT void libmeteor_loadbios(const void *data, unsigned size)
{
	AMeteor::_memory.LoadBios((const uint8_t*)data, size);
}

EXPORT uint8_t *libmeteor_getmemoryarea(int which)
{
	if (which < 7)
		return AMeteor::_memory.GetMemoryArea(which);
	else if (which == 7)
		return AMeteor::_io.GetIoPointer();
	else
		return NULL;
}

EXPORT int libmeteor_loadsaveram(const void *data, unsigned size)
{
	return AMeteor::_memory.LoadCart((const uint8_t*)data, size);
}

EXPORT int libmeteor_savesaveram(void **data, unsigned *size)
{
	return AMeteor::_memory.SaveCart((uint8_t **)data, size);
}

EXPORT void libmeteor_savesaveram_destroy(void *data)
{
	AMeteor::_memory.SaveCartDestroy((uint8_t *)data);
}

EXPORT int libmeteor_hassaveram()
{
	return AMeteor::_memory.HasCart();
}

EXPORT void libmeteor_clearsaveram()
{
	AMeteor::_memory.DeleteCart();
}

EXPORT int libmeteor_savestate(void **data, unsigned *size)
{
	if (!data || !size)
		return 0;

	std::ostringstream ss = std::ostringstream(std::ios_base::binary);
	AMeteor::SaveState(ss);

	std::string s = ss.str();

	void *ret = std::malloc(s.size());
	if (!ret)
		return 0;
	std::memcpy(ret, s.data(), s.size());
	*data = ret;
	*size = s.size();
	return 1;
}

EXPORT void libmeteor_savestate_destroy(void *data)
{
	std::free(data);
}

EXPORT int libmeteor_loadstate(const void *data, unsigned size)
{
	std::istringstream ss = std::istringstream(std::string((const char*)data, size), std::ios_base::binary);
	return AMeteor::LoadState(ss);
}

// TODO: cartram memory domain, cartram in system bus memory domain
EXPORT uint8_t libmeteor_peekbus(uint32_t addr)
{
	return AMeteor::_memory.Peek8(addr);
}

EXPORT void libmeteor_writebus(uint32_t addr, uint8_t val)
{
	AMeteor::_memory.Write8(addr, val);
}

int slcallbackline = 400;
void (*slcallback)() = NULL;

EXPORT void libmeteor_setscanlinecallback(void (*callback)(), int scanline)
{
	if (!callback)
		slcallbackline = 400;
	else
		slcallbackline = scanline;
	slcallback = callback;
}

void scanlinecallback_bizhawk()
{
	if (slcallback)
		slcallback();
}

EXPORT void libmeteor_getregs(int *dest)
{
	AMeteor::_cpu.UpdateCpsr();	
	for (int i = 0; i < 16; i++)
		dest[i] = AMeteor::_cpu.Reg(i);
	dest[16] = AMeteor::_cpu.Cpsr().dw;
	dest[17] = AMeteor::_cpu.Spsr().dw;
}
