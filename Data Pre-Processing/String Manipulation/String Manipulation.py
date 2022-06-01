# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 00:21:09 2021

@author: Karuna Singh
"""
# 1.
s1 = "Grow Gratitude"
# a)
letter = s1[0]
print(letter)

# b)
len(s1)

# c)
print(s1.count('G'))

**********************************************************************************

# 2.

s2 = "Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else."
print(s2.count(''))

**********************************************************************************

# 3.
s3 = "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"
#a)
print(s3[0:1])

#b)
print(s3[0:3])

#c)
print(s3[-3:])

**********************************************************************************

# 4.

s4 = "stay positive and optimistic"
spl = s4.split(' ')
spl

#a)
s4.startswith('H')
s4.endswith('d')
s4.endswith('c')
            
**********************************************************************************

# 5.

print("🪐 " * 108)

**********************************************************************************

# 6. doing in Python as not trained on R yet
print(" o" * 108)

**********************************************************************************

# 7.

s7 = "Grow Gratitude"
s7.replace('Grow','Growth of')

**********************************************************************************

# 8.

s8 = ".elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A"
print(''.join(reversed(s8)))

