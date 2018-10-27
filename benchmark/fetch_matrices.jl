using MatrixDepot

const spd_small = ("HB/bcsstk01", "FIDAP/ex5", "HB/494_bus", "HB/plat362")
const spd_med = ("HB/bcsstk09", "Bates/Chem97ZtZ", "HB/plat1919",
                      "Boeing/nasa1824", "Oberwolfach/t3dl_e")
const spd_large = ("GHS_psdef/torsion1", "Cannizzo/sts4098", "GHS_psdef/obstclae")
const matrix_path = dirname(pathof(MatrixDepot))

function get_matrices(matrices)
    for matrix in matrices
        try
            # throws an error if already downloaded
            matrixdepot(matrix, :get)
        catch
        end
    end
end

get_matrices(spd_small)
get_matrices(spd_med)
get_matrices(spd_large)
