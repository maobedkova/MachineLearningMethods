class Student:
    def __init__(self, name, points=0):
        self.name = name
        self.points = points
        self.hw = False

    def hand_in_hw(self):
        self.hw = True

    def evaluate(self):
        if self.hw:
            self.points += 10
            self.hw = False

if __name__ == "__main__":
    st = Student("mo")
    st.hand_in_hw()
    st.evaluate()
    st.hand_in_hw()
    st.evaluate()
    if st.name == "mo" and st.points == 20:
        print("Class exercise:", True)