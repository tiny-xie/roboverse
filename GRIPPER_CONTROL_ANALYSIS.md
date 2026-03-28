# RoboVerse 夹爪无法夹持物体的根本原因

## 问题诊断

### 为什么 LIBERO 和 ManiSkill 能夹起罐头，但 RoboVerse 不能？

**答案：控制器架构差异 — 闭环 vs 开环控制**

---

## 三者控制器对比

### 1️⃣ **ManiSkill: 显式PD反馈控制**

```python
# mani_skill/agents/controllers/pd_joint_pos.py (lines 113-119)
@dataclass
class PDJointPosControllerConfig(ControllerConfig):
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]          # ← kp 参数
    damping: Union[float, Sequence[float]]            # ← kd 参数
    force_limit: Union[float, Sequence[float]] = 1e10
```

**执行流程：**
```
agent.set_action(target_q)
  ↓
PDJointPosController.set_action()
  ↓
for each joint:
    joint.set_drive_properties(kp, kd, force_limit)  ← 关键！主动设置PD反馈
    joint.set_joint_drive_targets(target_q)
  ↓
Sapien物理引擎计算: τ = kp*(q_target - q_current) + kd*(v_target - v_current)
  ↓
即使夹爪接触物体，PD反馈会持续施加力量直到达到目标位置或达到force_limit
```

**关键特点：**
- ✅ **闭环反馈**：持续监测位置误差和速度，实时调整力量
- ✅ **力限制保护**：force_limit=100 防止过度施力
- ✅ **在Sapien级别实现**：物理引擎直接应用PD控制

---

### 2️⃣ **LIBERO/robosuite: 速度积分控制**

```python
# robosuite/robosuite/models/grippers/panda_gripper.py (lines 35-50)
def format_action(self, action):
    """
    Maps continuous action into gripper command
    """
    assert len(action) == self.dof  # dof=1 for PandaGripper
    self.current_action = np.clip(
        self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action),
        -1.0, 1.0
    )
    return self.current_action  # ← 返回**累积的**动作值
```

**执行流程：**
```
agent.set_action(gripper_cmd)  # e.g., [0] (close) or [1] (open)
  ↓
PandaGripper.format_action()
  ↓
current_action += speed * np.sign(gripper_cmd)  # 累积速度
  ↓
将累积的动作量（-1到1）映射到关节位置（0.0到0.04）
  ↓
MuJoCo actuator 应用 kp=1000 (在XML中定义) 逼近目标位置
  ↓
如果夹爪接触物体被卡住，仍在持续施加力（kp=1000的反馈）
```

**关键特点：**
- ✅ **平滑性**：速度积分避免突兀的位置跳跃
- ✅ **隐式力反馈**：依赖MuJoCo XML中的 `kp=1000` 和 `damping=100`
- ✅ **robosuite控制频率**：20Hz（合理）

---

### 3️⃣ **RoboVerse: 纯位置控制（无显式反馈）**

```python
# sim/mujoco/mujoco.py (lines 920-935)
def set_dof_targets(self, actions):
    """RoboVerse直接写入MuJoCo ctrl值"""
    for i in range(num_dofs):
        joint_name = joint_names[i]
        actuator_id = physics.model.actuator(...).id
        physics.data.ctrl[actuator_id] = actions[i]  # ← 直接设置ctrl=目标位置
        # 没有计算τ = kp*(q_target-q_current)！
```

**执行流程：**
```
agent.set_action([..., 0.0, 0.0])  # 夹爪目标位置：闭合
  ↓
process_gripper_command()  # ik_solver.py 将二值映射到 [0.0, 0.04]
  ↓
physics.data.ctrl[finger_actuator_1] = 0.0
physics.data.ctrl[finger_actuator_2] = 0.0
  ↓
MuJoCo取决于XML中的actuator定义...（问题来了！）
  ↓
❌ 如果XML中kp不够高，或者RoboVerse中stiffness参数没有被正确映射到XML，
   MuJoCo只是"尝试"达到位置0.0，但罐头一阻挡就放弃了
```

**关键问题：**
- ❌ **纯开环**：设置目标位置后"放手"，没有主动反馈
- ❌ **依赖XML配置**：stiffness参数可能没有被正确反映到MuJoCo actuator中
- ❌ **控制频率低**：decimation=15 → 33Hz（相比robosuite的20Hz都不算低，但对于细致控制还是有点慢

）
- ❌ **缺少速度积分**：没有平滑的闭合动作，直接跳到目标位置

---

## 问题根源总结

| 方面 | ManiSkill | robosuite | RoboVerse |
|------|-----------|-----------|-----------|
| 反馈机制 | 显式PD（代码计算τ） | 隐式PD（MuJoCo actuator）| 纯开环（XML被动） |
| 反馈闭合 | ✅ 即使被阻挡也继续施力 | ✅ MuJoCo kp反馈 | ❌ 没有主动反馈机制 |
| 控制信号 | target_q + set_drive_properties | target_q + 积分速度 | ctrl=q（直接） |
| 夹爪稳定性 | 极好 | 好 | 差 |

---

## 解决方案

### 方案A：修改RoboVerse MuJoCo XML（短期）

确保actuator的 `kp` 值足够高：

```xml
<actuator>
    <!-- 原配置 -->
    <position name="finger1_ctrl" 
              joint="panda_finger_joint1" 
              ctrlrange="0.0 0.04"
              kp="1000" />   <!-- ← 这个值必须大 -->
    
    <!-- 如果还是不行，改为: -->
    <position name="finger1_ctrl" 
              joint="panda_finger_joint1" 
              ctrlrange="0.0 0.04"
              kp="5000" />   <!-- ← 大幅增加 -->
</actuator>
```

**验证命令：**
```bash
grep -n "kp\|damping" roboverse_data/robots/franka/mjcf/panda.xml
```

---

### 方案B：在Python中实现显式PD控制（推荐）✅

**已创建文件：**
- `release/metasim/example/example_pack/pd_gripper_controller.py` - PDGripperController 类
- `scripts/advanced/replay_with_pd_gripper.py` - 集成示例

**使用方法：**

```python
from pd_gripper_controller import PDGripperController

# 初始化
gripper_pd = PDGripperController(
    joint_names=["panda_finger_joint1", "panda_finger_joint2"],
    kp=2000.0,   # Panda最大主观握力 ~140N，所以kp=2000很合理
    kd=400.0,    # 足够的阻尼避免振荡
    max_force=100.0,
)

# 在每个控制步
gripper_pd.set_target(gripper_command)  # Binary: 0 or 1
torques = gripper_pd.compute_torques(
    current_positions={"panda_finger_joint1": q1, "panda_finger_joint2": q2},
    current_velocities={"panda_finger_joint1": v1, "panda_finger_joint2": v2},
    dt=0.002
)

# torques 现在给出了应该施加的力量，即使夹爪被物体卡住
```

---

### 方案C：以LIBERO的方式修改RoboVerse（最根本）

将RoboVerse改为使用速度积分而不是直接位置控制：

```python
# 修改 process_gripper_command() 或新增 GripperActionAccumulator

class GripperActionAccumulator:
    def __init__(self, speed=0.2):
        self.current_action = 0.0
        self.speed = speed
    
    def update(self, gripper_cmd):
        self.current_action = np.clip(
            self.current_action + self.speed * np.sign(gripper_cmd),
            -1.0, 1.0
        )
        # 返回平滑的闭合运动而非直接位置
        target_q = 0.02 + self.current_action * 0.02
        return target_q
```

---

## 立即行动建议

### 优先级 1（最快）：验证XML中的kp值
```bash
# 检查 RoboVerse 使用的 panda.xml
find ~/robot_manipulation -name "panda.xml" -path "*/roboverse*" -exec grep "kp" {} +
```

### 优先级 2（推荐）：使用已创建的PDGripperController
```python
# 在你的 replay 脚本中集成
python scripts/advanced/replay_with_pd_gripper.py
```

### 优先级 3（长期）：修改RoboVerse核心
- 在 `ik_solver.py` 的 `process_gripper_command()` 中添加速度积分
- 或者在 `mujoco.py` 的 `set_dof_targets()` 中添加显式PD逻辑

---

## 参考文献

- **ManiSkill PD Controller**: `/mani_skill/agents/controllers/pd_joint_pos.py` (lines 113-290)
- **robosuite Gripper**: `/robosuite/robosuite/models/grippers/panda_gripper.py` (lines 30-60)
- **robosuite XML**: `/robosuite/robosuite/models/assets/grippers/panda_gripper.xml`
- **RoboVerse MuJoCo Backend**: `/sim/mujoco/mujoco.py` (lines 920-960)
- **RoboVerse Gripper Processing**: `/release/metasim/utils/ik_solver.py` (lines 151-187)
