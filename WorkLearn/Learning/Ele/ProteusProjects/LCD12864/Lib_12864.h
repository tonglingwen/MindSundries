#ifndef __LIB_12864__
#define __LIB_12864__
#include"Lib_common.h"
/***
*DATA.0:DB0
*DATA.1:DB1
*DATA.2:DB2
*DATA.3:DB3
*DATA.4:DB4
*DATA.5:DB5
*DATA.6:DB6
*DATA.7:DB7
*CONTROL.0:RS
*CONTROL.1:RW
*CONTROL.2:E
*CONTROL.3:CS1
*CONTROL.4:CS2
***/

sfr DATA=0x90;
sfr CONTROL=0xB0;


void Set_Pixel(uint8_t x,uint8_t y,uint8_t value,uint8_t issubmit);

void Set_Char16_16(uint8_t x,uint8_t y,uint8_t* value,uint8_t issubmit);

void Set_Char8_8(uint8_t x,uint8_t y,uint8_t* value,uint8_t issubmit);

void Set_Char12_12(uint8_t x,uint8_t y,uint8_t* value,uint8_t issubmit);

void Set_Char16_8(uint8_t x,uint8_t y,uint8_t* value,uint8_t issubmit);

void Set_uint8(uint8_t x,uint8_t y,uint8_t value,uint8_t issubmit,uint8_t b);

void Submit();

void InitLCD();

uint8_t Write(uint8_t d);

void Clear();

void Write_Command(uint8_t command);

void Read_Command(bool* bf,uint8_t* ac);

uint8_t Write_Data(uint8_t d);

void Read_Data(uint8_t* d);

void Submit_Write();

void Submit_Read();

void SetPosition(uint8_t row_x,uint8_t page_y);

void commondover();

void foucus_left();

void foucus_right();

void ac_left();

void dac_left();

void ac_right();

void dac_right();

void set_row(uint8_t r);//0~63

void set_page(uint8_t p);//0~7

void set_line(uint8_t l);//0~63

#endif