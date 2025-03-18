import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

class TrussSolver:
    def __init__(self, joints, members, supports, loads):
        self.joints = joints  
        self.members = members  
        self.supports = supports  
        self.loads = loads  
        self.force_matrix = None
        self.solution = None

    def solve(self):
        num_members = len(self.members)
        num_joints = len(self.joints)
        equations = []
        rhs = []
        
        joint_forces = {joint: [] for joint in self.joints}
        
        for i, (joint1, joint2) in enumerate(self.members):
            x1, y1 = self.joints[joint1]
            x2, y2 = self.joints[joint2]
            length = np.hypot(x2 - x1, y2 - y1)
            cos_theta = (x2 - x1) / length
            sin_theta = (y2 - y1) / length
            
            joint_forces[joint1].append((i, cos_theta, sin_theta))
            joint_forces[joint2].append((i, -cos_theta, -sin_theta))
        
        for joint, forces in joint_forces.items():
            row_x = np.zeros(num_members)
            row_y = np.zeros(num_members)
            
            for i, cos_theta, sin_theta in forces:
                row_x[i] = cos_theta
                row_y[i] = sin_theta
            
            force_x, force_y = self.loads.get(joint, (0, 0))
            support_x, support_y = self.supports.get(joint, (0, 0))
            
            equations.append(row_x)
            rhs.append(-force_x - support_x)
            equations.append(row_y)
            rhs.append(-force_y - support_y)
        
        self.force_matrix = np.array(equations)
        rhs = np.array(rhs)
        
        self.solution = np.linalg.solve(self.force_matrix, rhs)
        return self.solution

    def get_results(self):
        if self.solution is None:
            return {"error": "No solution found. Run solve() first."}
        
        results = []
        for i, force in enumerate(self.solution):
            state = "Tension" if force > 0 else "Compression"
            results.append({"member": self.members[i], "force": round(abs(force), 2), "state": state})
        return results

@app.route('/solve_truss', methods=['POST'])
def solve_truss():
    data = request.get_json()
    joints = data.get('joints', {})
    members = data.get('members', [])
    supports = data.get('supports', {})
    loads = data.get('loads', {})
    
    solver = TrussSolver(joints, members, supports, loads)
    solver.solve()
    results = solver.get_results()
    return jsonify(results)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
