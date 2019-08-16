#include<reg51.h>
#include<stdio.h>

typedef unsigned char uint8_t;

void initlizeprintf(void);

void delay500ms(void);

void my_print(char* c);

void Timer0_Init();

void accounter(int v);

void control0();

void control1();

void control2();

void control3();

int rem=0;
int h=0;
uint8_t currentNum[4]={0,0,0,0};
void main()
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

