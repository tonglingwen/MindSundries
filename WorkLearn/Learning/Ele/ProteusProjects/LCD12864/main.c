#include<reg51.h>
#include<stdio.h>
#include"Lib_12864.h"



void initlizeprintf(void);

void delay500ms(void);

void my_print(char* c);

void Timer0_Init();

void accounter(int v);

void control0();

void control1();

void control2();

void control3();

void counter_model();

void INT0_initilize();

int rem=0;
int h=0;
uint8_t currentNum[4]={0,0,0,0};
uint8_t code clear_hz[]={0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
uint8_t code HZ_0[]={0x00,0x00,0x80,0x60,0xF8,0x07,0x20,0x10,0x18,0xA7,0x44,0xA4,0x14,0x0C,0x00,0x00,
0x00,0x01,0x00,0x00,0xFF,0x02,0x02,0x01,0x21,0x24,0x44,0x48,0x91,0x01,0x02,0x02};
void main()
{
	int i=0;
	int t=0;
	int x=0;
	int y=0;
InitLCD();
	
	while(1)
	{
		
		i=0;
		Set_Char16_16(3,3,HZ_0,1);
		for(;i<10000;i++);
		
		i=0;
		 Set_Char16_16(3,3,clear_hz,1);
		for(;i<10000;i++);
	}
	/*
  foucus_left();
  SetPosition(0,0);
	for(;i<32;i++)
	{
		Write(HZ_[i]);
	}
i=0;
	for(;i<32;i++)
	{
		Write(HZ_0[i]);
	}
	i=0;
		for(;i<32;i++)
	{
	t=0;
	  for(;t<10000;t++);
		Write(HZ_1[i]);
	}
  		i=0;
		for(;i<32;i++)
	{
	t=0;
	  for(;t<10000;t++);
		Write(HZ_2[i]);
	}
	
	  		i=0;
		for(;i<32;i++)
	{
	t=0;
	  for(;t<10000;t++);
		Write(HZ_3[i]);
	}
	Clear();
	*/
	//Show();
	//set_page(0);
	//Write_Command(0x0F);
	i=0;
	t=0;
	for(;i<255;i++)
	{
		set_line(i);
		for(;t<10000;t++);
	//Write_Data(0x05);
		t=0;
	}
	//Clear();
	while(1);
}

void counter_model()
{
	Timer0_Init();
	//initlizeprintf();

	P2=0x0;
	

while(1)
	{
		//P2=0x0;
		//P2=0x1;
		
	
  h=TL0|(TH0<<8);
	accounter(h-rem);
	rem=h;
	//delay500ms();
}
}

void accounter(int v)
{
	if(v!=1)
		return;
	currentNum[3]++;
	if(currentNum[3]==10)
	{
		currentNum[3]=0;
		currentNum[2]++;
		if(currentNum[2]==10)
		{
			currentNum[2]=0;
			currentNum[1]++;
			if(currentNum[1]==10)
			{
				currentNum[1]=0;
				currentNum[0]++;
				if(currentNum[0]==10)
				{
					currentNum[0]=0;
				}
			}
		}
	}
	
	control0();
	control1();
	control2();
	control3();
}

void control0()
{
	int t=0;
	uint8_t num = currentNum[3];
	uint8_t val_p=0;
	uint8_t tem=0;
	int i=0;
	for(;i<4;i++)
	{
		tem=P2;
		tem&=0xFC;
		val_p=tem|((num&0x01)<<1);
		P2=val_p;
		P2|=0x01;
		num=num>>1;
	}
}

void control1()
{
	int t=0;
	uint8_t num = currentNum[2];
	uint8_t val_p=0;
	uint8_t tem=0;
	int i=0;
	for(;i<4;i++)
	{
		tem=P2;
		tem&=0xF3;
		val_p=tem|((num&0x01)<<3);
		P2=val_p;
		P2|=(0x01<<2);
		num=num>>1;
	}
}

void control2()
{
	int t=0;
	uint8_t num = currentNum[1];
	uint8_t val_p=0;
	uint8_t tem=0;
	int i=0;
	for(;i<4;i++)
	{
		tem=P2;
		tem&=0xCF;
		val_p=tem|((num&0x01)<<5);
		P2=val_p;
		P2|=(0x01<<4);
		num=num>>1;
	}
}

void control3()
{
	int t=0;
	uint8_t num = currentNum[0];
	uint8_t val_p=0;
	uint8_t tem=0;
	int i=0;
	for(;i<4;i++)
	{
		tem=P2;
		tem&=0x3F;
		val_p=tem|((num&0x01)<<7);
		P2=val_p;
		P2|=(0x01<<6);
		num=num>>1;
	}
}


void delay500ms(void)
{
	unsigned char a,b,c;
	for(c=23;c>0;c--)
		for(b=152;b>0;b--);
			//for(a=70;a>0;a--);
}

void initlizeprintf(void)
{
SCON= 0x50;   //设置串口工作方式
	
	
TMOD |= 0x20;  //设置计数器工作方式
TH1= 0xf3; //计数器初值 波特率2400
TR1= 1;    //启动计数器
TI= 1;     
}

void my_print(char* c)
{
	while(*c!=0)
	{
		TI=0;
		SBUF=*c;
		while(!TI);
		c++;
	}
}

void Timer0_Init() //初始化定时器
{
	/*
   TMOD = 0x05;		//
	 TH0=0xFF;        //重新给定初值
   TL0=245; 
	 EA=1;            //总中断打开
   ET0=1;           //定时器中断打开
   TR0 = 1;	//启动定时器0
	*/
	
	TMOD = 0x5;     //使用模式1，16位计数器，使用"|"符号可以在使用多个定时器时不受影响          

 //TH1=(65536-3000)/256;        //给定初值

 //TL1=245;         //从245计数到255 

 //EA=1;            //总中断打开

 //ET1=1;           //定时器中断打开

 TR0=1;           //定时器开关打开
}


void INT0_initilize()
{
 EA=1;//打开总中断开关
 EX0=1;//开外部中断0
 IT0=1;//设置外部中断的触发方式
}

void INT0_handler()interrupt 0
{
	if(P2==1)
		P2=0;
	else
		P2=1;
}

void Timer1_isr(void) interrupt 3
{
	P2=P2<<1;
	if(P2==0x00)
		P2=0x00;
	TH1=0xFF;        //重新给定初值

 TL1=245; 
 //printf("shui:\n");
	/*
 TH0=0xFF;        //重新给定初值
 TL0=245; 
	*/
}

