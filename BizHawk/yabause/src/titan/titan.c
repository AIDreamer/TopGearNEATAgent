/*  Copyright 2012 Guillaume Duhamel

    This file is part of Yabause.

    Yabause is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Yabause is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Yabause; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
*/

#include "titan.h"

#include <stdlib.h>

/* private */
typedef u32 (*TitanBlendFunc)(u32 top, u32 bottom);

static struct TitanContext {
   int inited;
   u32 * vdp2framebuffer[8];
   u32 * linescreen[4];
   int vdp2width;
   int vdp2height;
   TitanBlendFunc blend;
} tt_context = {
   0,
   { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
   { NULL, NULL, NULL, NULL },
   320,
   224
};

#if defined WORDS_BIGENDIAN
static INLINE u32 TitanFixAlpha(u32 pixel) { return ((((pixel & 0x3F) << 2) + 0x03) | (pixel & 0xFFFFFF00)); }

static INLINE u8 TitanGetAlpha(u32 pixel) { return pixel & 0x3F; }
static INLINE u8 TitanGetRed(u32 pixel) { return (pixel >> 8) & 0xFF; }
static INLINE u8 TitanGetGreen(u32 pixel) { return (pixel >> 16) & 0xFF; }
static INLINE u8 TitanGetBlue(u32 pixel) { return (pixel >> 24) & 0xFF; }
static INLINE u32 TitanCreatePixel(u8 alpha, u8 red, u8 green, u8 blue) { return alpha | (red << 8) | (green << 16) | (blue << 24); }
#else
static INLINE u32 TitanFixAlpha(u32 pixel) { return ((((pixel & 0x3F000000) << 2) + 0x03000000) | (pixel & 0x00FFFFFF)); }

static INLINE u8 TitanGetAlpha(u32 pixel) { return (pixel >> 24) & 0x3F; }
static INLINE u8 TitanGetRed(u32 pixel) { return (pixel >> 16) & 0xFF; }
static INLINE u8 TitanGetGreen(u32 pixel) { return (pixel >> 8) & 0xFF; }
static INLINE u8 TitanGetBlue(u32 pixel) { return pixel & 0xFF; }
static INLINE u32 TitanCreatePixel(u8 alpha, u8 red, u8 green, u8 blue) { return (alpha << 24) | (red << 16) | (green << 8) | blue; }
#endif

static u32 TitanBlendPixelsTop(u32 top, u32 bottom)
{
   u8 alpha, ralpha, tr, tg, tb, br, bg, bb;

   alpha = (TitanGetAlpha(top) << 2) + 3;
   ralpha = 0xFF - alpha;

   tr = (TitanGetRed(top) * alpha) / 0xFF;
   tg = (TitanGetGreen(top) * alpha) / 0xFF;
   tb = (TitanGetBlue(top) * alpha) / 0xFF;

   br = (TitanGetRed(bottom) * ralpha) / 0xFF;
   bg = (TitanGetGreen(bottom) * ralpha) / 0xFF;
   bb = (TitanGetBlue(bottom) * ralpha) / 0xFF;

   return TitanCreatePixel(0x3F, tr + br, tg + bg, tb + bb);
}

static u32 TitanBlendPixelsBottom(u32 top, u32 bottom)
{
   u8 alpha, ralpha, tr, tg, tb, br, bg, bb;

   if ((top & 0x80000000) == 0) return top;

   alpha = (TitanGetAlpha(bottom) << 2) + 3;
   ralpha = 0xFF - alpha;

   tr = (TitanGetRed(top) * alpha) / 0xFF;
   tg = (TitanGetGreen(top) * alpha) / 0xFF;
   tb = (TitanGetBlue(top) * alpha) / 0xFF;

   br = (TitanGetRed(bottom) * ralpha) / 0xFF;
   bg = (TitanGetGreen(bottom) * ralpha) / 0xFF;
   bb = (TitanGetBlue(bottom) * ralpha) / 0xFF;

   return TitanCreatePixel(TitanGetAlpha(top), tr + br, tg + bg, tb + bb);
}

static u32 TitanBlendPixelsAdd(u32 top, u32 bottom)
{
   u32 r, g, b;

   r = TitanGetRed(top) + TitanGetRed(bottom);
   if (r > 0xFF) r = 0xFF;

   g = TitanGetGreen(top) + TitanGetGreen(bottom);
   if (g > 0xFF) g = 0xFF;

   b = TitanGetBlue(top) + TitanGetBlue(bottom);
   if (b > 0xFF) b = 0xFF;

   return TitanCreatePixel(0x3F, r, g, b);
}

static u32 TitanDigPixel(int * priority, int pos)
{
   u32 pixel = 0;
   u8 alpha;
   while((*priority > -1) && (! pixel))
   {
      pixel = tt_context.vdp2framebuffer[*priority][pos];
      (*priority)--;
   }
   tt_context.vdp2framebuffer[*priority + 1][pos] = 0;
   if (*priority == -1) return pixel;

   alpha = TitanGetAlpha(pixel);
   if (alpha < 0x3F)
   {
      u32 bottom = TitanDigPixel(priority, pos);
      pixel = tt_context.blend(pixel, bottom);
   }
   return pixel;
}

/* public */
int TitanInit()
{
   int i;

   if (! tt_context.inited)
   {
      for(i = 0;i < 8;i++)
      {
         if ((tt_context.vdp2framebuffer[i] = (u32 *)calloc(sizeof(u32), 704 * 512)) == NULL)
            return -1;
      }

      /* linescreen 0 is not initialized as it's not used... */
      for(i = 1;i < 4;i++)
      {
         if ((tt_context.linescreen[i] = (u32 *)calloc(sizeof(u32), 512)) == NULL)
            return -1;
      }

      tt_context.inited = 1;
   }

   for(i = 0;i < 8;i++)
      memset(tt_context.vdp2framebuffer[i], 0, sizeof(u32) * 704 * 512);

   for(i = 1;i < 4;i++)
      memset(tt_context.linescreen[i], 0, sizeof(u32) * 512);

   return 0;
}

int TitanDeInit()
{
   int i;

   for(i = 0;i < 8;i++)
      free(tt_context.vdp2framebuffer[i]);

   for(i = 1;i < 4;i++)
      free(tt_context.linescreen[i]);

   return 0;
}

void TitanSetResolution(int width, int height)
{
   tt_context.vdp2width = width;
   tt_context.vdp2height = height;
}

void TitanGetResolution(int * width, int * height)
{
   *width = tt_context.vdp2width;
   *height = tt_context.vdp2height;
}

void TitanPutBackHLine(s32 y, u32 color)
{
   u32 * buffer = tt_context.vdp2framebuffer[0] + (y * tt_context.vdp2width);
   int i;

   for (i = 0; i < tt_context.vdp2width; i++)
      buffer[i] = color;
}

void TitanPutLineHLine(int linescreen, s32 y, u32 color)
{
   if (linescreen == 0) return;

   {
      u32 * buffer = tt_context.linescreen[linescreen] + y;
      *buffer = color;
   }
}

void TitanPutPixel(int priority, s32 x, s32 y, u32 color, int linescreen)
{
   if (priority == 0) return;

   {
      int pos = (y * tt_context.vdp2width) + x;
      u32 * buffer = tt_context.vdp2framebuffer[priority] + pos;
      if (linescreen)
         color = TitanBlendPixelsTop(color, tt_context.linescreen[linescreen][y]);
      *buffer = color;
   }
}

void TitanPutHLine(int priority, s32 x, s32 y, s32 width, u32 color)
{
   if (priority == 0) return;

   {
      u32 * buffer = tt_context.vdp2framebuffer[priority] + (y * tt_context.vdp2width) + x;
      int i;

      for (i = 0; i < width; i++)
         buffer[i] = color;
   }
}

void TitanPutShadow(int priority, s32 x, s32 y)
{
   if (priority == 0) return;

   {
      int pos = (y * tt_context.vdp2width) + x;
      u32 * buffer = tt_context.vdp2framebuffer[priority] + pos;
      *buffer = *buffer ? TitanBlendPixelsTop(0x20000000, *buffer) : 0x20000000;
   }
}

void TitanRender(u32 * dispbuffer, int blend_mode)
{
   u32 dot;
   int i, p;

   if (blend_mode == TITAN_BLEND_BOTTOM)
      tt_context.blend = TitanBlendPixelsBottom;
   else if (blend_mode == TITAN_BLEND_ADD)
      tt_context.blend = TitanBlendPixelsAdd;
   else
      tt_context.blend = TitanBlendPixelsTop;

   for (i = 0; i < (tt_context.vdp2width * tt_context.vdp2height); i++)
   {
      p = 7;
      dot = TitanDigPixel(&p, i);
      if (dot)
      {
         dispbuffer[i] = TitanFixAlpha(dot);
      }
   }
}
