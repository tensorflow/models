#define uchar _utfuchar
#define ushort _utfushort
#define uint _utfuint
#define ulong _utfulong
#define vlong _utfvlong
#define uvlong _utfuvlong

typedef unsigned char		uchar;
typedef unsigned short		ushort;
typedef unsigned int		uint;
typedef unsigned long		ulong;

#define nelem(x) (sizeof(x)/sizeof((x)[0]))
#define nil ((void*)0)
