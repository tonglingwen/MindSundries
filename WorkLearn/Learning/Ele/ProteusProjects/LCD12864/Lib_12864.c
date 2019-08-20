#include"Lib_12864.h"
#include<reg51.h>




uint8_t xdata Screen[8][128];

uint8_t curr_X=0;
uint8_t curr_Y=0;

void InitLCD()
{
	int t=0;
	int i=0;
	for(;t<8;t++)
	{
		i=0;
		for(;i<128;i++)
		{
			Screen[t][i]=0x00;
		}
	}
	//CONTROL=Screen[0][0];
	
	
	Submit();
}

void Set_Pixel(uint8_t x,uint8_t y,uint8_t value,uint8_t issubmit)
{
	uint8_t page= y/8;
	uint8_t bi=y%8;
	uint8_t val= Screen[page][x];
	if(x>127||y>63)
		return;
	
  if(value==0)
		Screen[page][x]=val&(~(1<<bi));
	else
		Screen[page][x]=val|(1<<bi);
	if(issubmit==1)
		Submit();
}

void Set_Char16_16(uint8_t x,uint8_t y,uint8_t* value,uint8_t issubmit)
{
	int i=0;
	for(;i<32;i++)
	{
		if(i<16)
			Set_uint8(x+i,y,value[i],0,8);
		else
			Set_uint8(x+i-16,y+8,value[i],0,8);
	}
	if(issubmit==1)
		Submit();
}

void Set_Char8_8(uint8_t x,uint8_t y,uint8_t* value,uint8_t issubmit)
{
	int i=0;
	for(;i<8;i++)
			Set_uint8(x+i,y,value[i],0,8);
	if(issubmit==1)
		Submit();
}

void Set_Char12_12(uint8_t x,uint8_t y,uint8_t* value,uint8_t issubmit)
{
		int i=0;
	for(;i<24;i++)
	{
		if(i<12)
			Set_uint8(x+i,y,value[i],0,8);
		else
			Set_uint8(x+i-12,y+8,value[i],0,4);
	}
	if(issubmit==1)
		Submit();
}

void Set_Char16_8(uint8_t x,uint8_t y,uint8_t* value,uint8_t issubmit)
{
	int i=0;
	for(;i<16;i++)
	{
		if(i<8)
			Set_uint8(x+i,y,value[i],0,8);
		else
			Set_uint8(x+i-8,y+8,value[i],0,8);
	}
	if(issubmit==1)
		Submit();
}

void Set_uint8(uint8_t x,uint8_t y,uint8_t value,uint8_t issubmit,uint8_t b)
{
	int i=0;
	uint8_t tem[8];
	for(;i<b;i++)
	{
		if(value&0x01>0)
			tem[i]=1;
		else
			tem[i]=0;
		value=value>>1;
	}
	i=0;
	for(;i<b;i++)
		Set_Pixel(x,y+i,tem[i],0);
	if(issubmit==1)
		Submit();
}

void Submit()
{
		int i;
	int t=0;
	for(;t<8;t++)
	{
		i=0;

		for(;i<128;i++){
		  SetPosition(i,t);
			Write_Data(~Screen[t][i]);
		}
		//Screen[t][i]
	}
	/*
	int t=0;
	int i=0;
	int k=0;
	SetPosition(0,0);
	autoline=1;
	for(;t<8;t++)
	{
		i=0;
		for(;i<128;i++)
		{
				k=0;
	  for(;k<10000;k++);
			Write(Screen[t][i]);
		}
	}
	*/
}

uint8_t Write(uint8_t d)
{
	if(Write_Data(d)==0)
	{
		if((CONTROL&0x08)==0)
		{
			foucus_right();
			set_line(0);
		}else
			return 0;
		Write_Data(d);
	}
	return 1;
}

void Clear()
{
	int i;
	int t=0;
	for(;t<8;t++)
	{
		i=0;
	  SetPosition(0,t);
		for(;i<128;i++)
			Write(0xFF);
	}
	InitLCD();
}

void Write_Command(uint8_t command)
{
	  CONTROL&=0xFD;
	  CONTROL&=0xFE;
	  DATA=command;
	  Submit_Write();
}

void Read_Command(bool* bf,uint8_t* ac)
{
		uint8_t t=0;
		CONTROL|=0x02;
		CONTROL&=0xFE;
		t=DATA;
		*ac=(t&0x7F);
		*bf=(t&0x80)>0;
}

uint8_t Write_Data(uint8_t d)
{
	  if(curr_X==64)
			return 0;
	  CONTROL&=0xFD;
		CONTROL|=0x01;
		DATA=d;
	  Submit_Write();
		curr_X++;
		return 1;
}

void Read_Data(uint8_t* d)
{
		CONTROL|=0x02;
		CONTROL|=0x01;
	  *d=DATA;
}

void Submit_Write()
{
	int i=0;
	CONTROL&=0xFB;
	for(;i>1000;i++);
	CONTROL|=0x04;
}

void Submit_Read()
{
	CONTROL|=0x04;
}

void SetPosition(uint8_t line_x,uint8_t page_y)
{
	if(page_y<0||page_y>7||line_x<0||line_x>127)
		return;
	if(line_x>63)
	{
		foucus_right();
		line_x=line_x-64;
	}
	else
	{
		foucus_left();
	}
	set_page(page_y);
  set_line(line_x);
}

void commondover()
{
		uint8_t t=0;
		CONTROL|=0x02;
		CONTROL&=0xFE;
		while((DATA&0x80)>0);
}

void foucus_left()
{
	ac_left();
	dac_right();
}

void foucus_right()
{
	ac_right();
	dac_left();
}

void ac_left()
{
	CONTROL&=0xF7;
}

void dac_left()
{
	CONTROL|=0x08;
}


void ac_right()
{
	CONTROL&=0xEF;
}


void dac_right()
{
	CONTROL|=0x10;
}

void set_row(uint8_t r)
{
	if(r>63)
		return;
	Write_Command(0xC0|r);
}

void set_page(uint8_t p)
{
		if(p>7)
			return;
		Write_Command(0xB8|p);
		curr_Y=p;
}

void set_line(uint8_t l)
{
		if(l>64)
			return;
		Write_Command(0x40|l);
		curr_X=l;
}