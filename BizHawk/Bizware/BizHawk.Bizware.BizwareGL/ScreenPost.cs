using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Linq;
using System.Text;

namespace BizHawk.Bizware.BizwareGL
{
    [DataContract]
    public unsafe class ScreenPost
    {
        // check http://stackoverflow.com/questions/9145667/how-to-post-json-to-the-server
        [DataMember]
        public int Width = 256;

        [DataMember]
        public int Height = 224;

        [DataMember]
        public int[] PixelsBG0 = new int[256 * 224];

        [DataMember]
        public int[] PixelsBG3 = new int[256*224];

        [DataMember]
        public int[] PixelsOBJ3 = new int[256*224];

        [DataMember]
        public int[] Pixels = new int[256 * 224];
        
        [DataMember]
        public uint Speed;

        [DataMember]
        public uint Nitro;

        [DataMember]
        public uint Distance;

        [DataMember]
        public uint Time;

        [DataMember]
        public uint Rank;

        [DataMember]
        public uint Lap;

        public ScreenPost() { }



    }
}
