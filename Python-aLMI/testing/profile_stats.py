# extract stats from a "profile file"
import sys
import pstats

print(sys.argv[0], "starting ...")
print("args =", sys.argv)

if len(sys.argv) < 2:
    print("usage:  python " + sys.argv[0] + " PROFILE_FILE [SORT_TYPE]")
    sys.exit(1)

pstats_file = sys.argv[1]
if len(sys.argv) > 2:
    sort_type = sys.argv[2]
else:
    sort_type = 'time'

p=pstats.Stats(pstats_file)		# load the profile stats

p.sort_stats(sort_type).print_stats(10)		# most time-consuming
p.print_stats('SDreaddata')			# reading values from input file
p.print_stats('writeValues')			# writing values to output file
# p.print_stats('lstsq')			# lstsq
p.print_stats('pyhdf\._hdfext\.')		# HDF
# p.print_stats('netCDF4\.Dataset')		# NetCDF4 (only used by SIOP_sets_load)
