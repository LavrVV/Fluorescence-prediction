import pydicom
f = './mrt/new/аболмазов р в/1.2.840.113619.2.312.6945.1127356.11178.1471437144.605.dcm'
encodings = ['ANSI', 'ISO-8859-5' ,'ISO-8859-1']
for e in encodings:
    try:
        ff = open(f,'rb')
        ds = pydicom.dcmread(f)
        break
    finally:
        ff.close()
sex = ds[(0x10, 0x40)].value
sex = 1 if sex == 'M' else 0
age = ds[(0x10, 0x1010)].value
age = int(age[:-1])
weight = ds[(0x10, 0x1030)].value
weight = float(weight)
ds = ds.pixel_array
print('ok')
