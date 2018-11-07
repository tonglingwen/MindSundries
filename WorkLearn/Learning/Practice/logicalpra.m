l=load('C:\Users\25285\Desktop\ss\save.txt');
l0=[0,0,0];
l1=[0,0,0];
l2=[0,0,0];
l3=[0,0,0];
for i=1:size(l)(1,1)
  if l(i,3)==0
    l0=[l0;l(i,:)];
  endif
  if l(i,3)==1
    l1=[l1;l(i,:)];
  endif
  if l(i,3)==2
    l2=[l2;l(i,:)];
  endif
  if l(i,3)==3
    l3=[l3;l(i,:)];
  endif
endfor
l0(1,:)=[]
l1(1,:)=[]
l2(1,:)=[]
l3(1,:)=[]

l0y=l0(:,3)
l1y=l1(:,3)
l0x=[ones(size(l0)(1,1),1),l0(:,1:2)]

syms k0 k1 k2;
pm=k0+k1+k2;
