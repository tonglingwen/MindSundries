syms k1 k2 x y lost;
syms py vy;
a= 8.47160;
b=-0.44731;
learnspeed=0.001;

datemat=load('C:\Users\25285\Desktop\sym\vrk.txt');

plot(datemat(:,1),datemat(:,2),'o')

py=datemat(:,2);

y=k1+k2*x;
kmat=[k1;k2];
xmat=datemat;
xmat(:,2)=xmat(:,1);
xmat(:,1)=1;
vy=xmat*kmat;
m=size(py)(1,1);
lost=1/(2*m)*sum((vy-py).^2);
k1diff=diff(lost,k1);
k2diff=diff(lost,k2);

#sd= pinv(xmat'*xmat)*xmat'*py 正规方程法
#lostval1=vpa(subs(lost,{'k1','k2'},[sd(1,1),sd(2,1)]))


for i=1:20
tema=a-learnspeed* subs(k1diff,{'k1','k2'},[a,b]);
temb=b-learnspeed* subs(k2diff,{'k1','k2'},[a,b]);
a=tema;
b=temb;
lostval=vpa(subs(lost,{'k1','k2'},[a,b]))
hold on
plot([0;30],double([a;a+30*b]))
end
vpa(a)
vpa(b)



