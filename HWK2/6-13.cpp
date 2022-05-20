#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

using Map = std::vector<std::vector<int>>;

int dir[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

bool check_valid(int x, int y, int m, int n) {
    return (x >= 0 && x < m && y >= 0 && y < n);
}

void print(std::vector<std::vector<int>>& map) {
    int m = map.size();
    int n = map[0].size();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << map[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int m, n;
    std::string path = "./map.txt";
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cout << "Cannot open file " << path << std::endl;
        return 0;
    }
    ifs >> m >> n;
    Map map_0(m, std::vector<int>(n));
    Map map_1(m, std::vector<int>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            ifs >> map_0[i][j];
        }
    }
    std::cout << "Init state: " << std::endl;
    print(map_0);
    Map* cur_map;
    Map* next_map;
    for (int k = 0; k < 5; ++k) {
        std::cout << "Iter " << k << ": " << std::endl;
        if (!(k % 2)) {
            cur_map = &map_0;
            next_map = &map_1;
        } else {
            cur_map = &map_1;
            next_map = &map_0;
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int live_cnt = 0;
                for (int d = 0; d < 8; ++d) {
                    int cur_i = i + dir[d][0];
                    int cur_j = j + dir[d][1];
                    if (check_valid(cur_i, cur_j, m, n) && (*cur_map)[cur_i][cur_j] == 1) {
                        live_cnt++;
                    }
                }
                if ((*cur_map)[i][j] == 0) {
                    if (live_cnt == 3) {
                        (*next_map)[i][j] = 1;
                    } else {
                        (*next_map)[i][j] = 0;
                    }
                } else {
                    if (live_cnt == 2 || live_cnt == 3) {
                        (*next_map)[i][j] = 1;
                    } else {
                        (*next_map)[i][j] = 0;
                    }
                }
            }
        }
        print(*next_map);
    }
    return 0;
}