# Technical Method: Bomberman DQN

Tài liệu này mô tả method của repo cho người mới: AI nhận gì từ game, học như thế nào, cách đánh giá, và tiêu chí để kết luận kết quả.

## 1. Bài toán

Repo dùng game Bomberman làm môi trường reinforcement learning (RL). Agent `dqn_agent` nhìn bàn cờ, chọn một trong sáu action, nhận reward từ engine và lặp lại quá trình đó để học.

```text
board state -> DQN chọn action -> game engine cập nhật -> reward + board state mới
                    ^                                      |
                    +----------- replay buffer ------------+
```

Sáu action là `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, `BOMB`. Data train online không phải file gán nhãn có sẵn: nó được sinh khi agent chơi game.

| File/thư mục | Vai trò |
|---|---|
| `environment.py` | Engine: bàn cờ, bom, nổ, điểm và kết thúc ván. |
| `bomber_rl/core.py` | State encoding, network, replay, safety, checkpoint. |
| `bomber_rl/agent.py` | Nối callback của engine với DQN learning loop. |
| `bomber_rl/runner.py` | Train, demos, pretrain, evaluation, curriculum. |
| `agent_code/*` | DQN agent và các đối thủ heuristic. |
| `config/*.yaml` | Tham số thí nghiệm tái lập được. |

## 2. Model nhìn thấy gì?

Model không dùng ảnh màn hình. Mỗi game state được đổi thành tensor `uint8` kích thước `C × 17 × 17`.

Baseline `config/dqn.yaml` có 7 kênh:

```text
walls, crates, coins, bombs + timers, explosions, self, opponents
```

`config/dqn_enhanced.yaml` thêm kênh danger map. Giá trị tại một ô là deadline sớm nhất ô đó có thể trúng blast. Blast ray đi theo bốn hướng, dừng ở tường và dừng sau crate. Điều này giúp model biết crate có thể che chắn nguy hiểm.

## 3. DQN học như thế nào?

Network trả sáu Q-value:

```text
Q(state, UP), Q(state, RIGHT), ..., Q(state, BOMB)
```

Q-value là reward tương lai ước lượng nếu chọn action đó. Khi train, epsilon-greedy cho phép agent đôi khi chọn action ngẫu nhiên để khám phá. Mỗi lần đi tạo transition:

```text
(observation, action, reward, next_observation, terminated, next_action_mask)
```

Transition được giữ trong replay buffer `uint8`. Model lấy batch từ replay và tối ưu Huber loss. Reward có phần cho coin, kill, crate, death, invalid action và survived round.

## 4. Baseline và enhanced pipeline

`config/dqn.yaml` được giữ nguyên làm vanilla-DQN baseline: state 7 kênh, uniform replay, target DQN chuẩn.

`config/dqn_enhanced.yaml` bật các thành phần sau:

| Thành phần | Tác dụng |
|---|---|
| Double DQN | Online network chọn action; target network định giá action đó, giảm Q-value quá lạc quan. |
| Dueling network | Tách giá trị state khỏi lợi thế của từng action. |
| Prioritized replay | TD error lớn được sample nhiều hơn; IS weight giảm sampling bias. |
| Danger map | Thêm tín hiệu trực tiếp về vùng blast. |
| Safe-action mask | Loại action chắc chắn chết nếu còn action an toàn. |
| Symmetry augmentation | Xoay/lật batch và remap action/mask; không lưu tám bản sao. |
| Demos + pretrain | Bắt đầu bằng việc bắt chước agent heuristic. |
| Curriculum | Đối thủ tăng độ khó theo số training step. |

Không có mục nào đảm bảo model mạnh hơn. Điều đó phải được chứng minh qua final test nhiều seed.

## 5. Action mask và safety

Legal action mask loại nước đi vào tường, crate, bom, đối thủ hoặc đặt bom khi không còn bom. Safe-action mask kiểm tra thêm đường thoát trước deadline blast.

Khi action là `BOMB`, agent chỉ được đặt bom nếu search tìm được đường thoát. Nếu mọi action đều bị đánh giá nguy hiểm, code fallback về legal mask thay vì trả vector toàn `false`; engine luôn nhận được action hợp lệ.

## 6. Đối thủ và curriculum

| Agent | Hành vi |
|---|---|
| `random_agent` | Đối thủ dễ, gần ngẫu nhiên. |
| `rule_based_agent` | Thu coin, phá crate và tránh bom bằng heuristic. |
| `aggressive_rule_based_agent` | Thiên về săn đối thủ. |
| `hard_rule_based_agent` | Ưu tiên `dqn_agent`, tránh blast path và chỉ đặt bom khi có exit. |

Enhanced curriculum mặc định:

```text
0–100k:     random + rule + rule
100k–200k:  rule + rule + rule
200k–300k:  hard + rule + rule
300k–400k:  hard + aggressive + rule
```

Khi đổi stage, runner lưu checkpoint rồi khởi tạo game với opponent set mới. Model, optimizer, replay, PER và RNG được restore.

## 7. Demonstrations và behavior cloning

`collect-demos` chạy agent rule trong chính engine và ghi NPZ shards. Mỗi transition có các trường:

```text
observation, action, reward, next_observation,
terminated, next_action_mask, episode_id, opponent_set
```

`pretrain` học mapping `observation -> action` bằng cross-entropy. Train/validation split thực hiện theo `episode_id` (xấp xỉ 90/10), không chia random từng transition. Lý do: state trong một ván tương quan rất mạnh; nếu xuất hiện ở cả hai phía thì validation sẽ lạc quan giả tạo.

Pretrain chỉ là điểm khởi tạo. Online RL vẫn cần tiếp tục train để tối ưu reward và vượt giới hạn heuristic.

## 8. Validation, final test và checkpoint selection

| Nhóm data | Có cập nhật model? | Mục đích |
|---|---|---|
| Online transitions | Có | Học Q-value. |
| Demo validation episodes | Không | Early stopping cho behavior cloning. |
| RL validation seeds | Không | Chọn `best.pt`. |
| Final test seeds | Không | Báo cáo cuối cùng. |

Enhanced config tách seed validation `1000–1009` và test `2000–2009`. Periodic evaluation chỉ dùng validation. `best.pt` được chọn theo thứ tự:

```text
win_rate cao hơn -> mean_return cao hơn -> death_rate thấp hơn
```

Final test không được dùng để chọn checkpoint. Kết quả JSON/CSV được tách theo `rule`, `hard`, và `mixed` opponent sets.

Mặc định có 10 validation và 10 test seeds. Trước khi nộp, nên tăng lên 50 seed cho mỗi nhóm và đánh giá baseline/enhanced trên đúng cùng seed, opponent set và train budget.

## 9. Resume và reproducibility

`latest.pt` được ghi atomically. Nó lưu:

```text
online model, target model, optimizer, AMP scaler,
replay buffer, PER priorities, RNG state,
curriculum stage, best validation metric, training step
```

Chạy lại cùng lệnh `train` sẽ resume. `pretrain` cũng resume từ `<output>.latest.pt`; dùng `--no-resume` để yêu cầu một run mới. `collect-demos` yêu cầu `--resume` để quét các NPZ hoàn chỉnh và tiếp tục bằng shard/episode kế tiếp. Nếu observation shape hoặc replay capacity khác checkpoint, repo báo lỗi rõ ràng thay vì âm thầm train sai.

## 10. Mục đích, cách thực hiện, kết quả kỳ vọng

### Mục đích

1. Xây dựng agent Bomberman học từ trải nghiệm game.
2. So sánh vanilla DQN với pipeline safety/PER/Double/Dueling/demos/curriculum.
3. Làm thí nghiệm tái lập được qua config, seed, log và checkpoint.

### Cách thực hiện

1. Chạy unit test và smoke test.
2. Collect demos từ hard/rule agents.
3. Pretrain behavior cloning.
4. Train baseline và enhanced với cùng budget, tối thiểu ba seed.
5. Chọn checkpoint bằng validation.
6. Test checkpoint bằng held-out seeds và cùng opponent sets.
7. Báo cáo trung bình và độ biến thiên qua các seed.

### Kết quả kỳ vọng

- Checkpoint resume không mất optimizer, replay hoặc training step.
- Danger/safe mask giảm các action nguy hiểm rõ ràng.
- Demos tạo hành vi hợp lệ ở giai đoạn đầu nhanh hơn.
- Curriculum tránh overfit vào một loại đối thủ.
- Báo cáo gồm `win_rate`, `mean_return`, `death_rate` theo từng opponent set.

Enhanced pipeline chỉ được kết luận tốt hơn baseline khi nó đạt kết quả tốt hơn trên cùng final test protocol. Nếu chưa có số liệu đó, báo cáo cần nói đây là method cần được kiểm chứng, không phải kết luận hiệu năng.

## 11. Thứ tự nên đọc code

1. `config/dqn.yaml` và `config/dqn_enhanced.yaml`.
2. `bomber_rl/core.py`.
3. `bomber_rl/agent.py`.
4. `bomber_rl/runner.py`.
5. `agent_code/hard_rule_based_agent/callbacks.py`.
6. `tests/test_core.py`.
