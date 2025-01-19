import React from 'react';
import { SafeAreaView, ScrollView, Text, View, Button, StyleSheet, TouchableOpacity } from 'react-native';

const IndexScreen = () => {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        {/* 제목 */}
        <View style={styles.header}>
          <Text style={styles.title}>시니어 헬스 커뮤니티</Text>
        </View>

        {/* 건강 정보 섹션 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>오늘의 건강 팁</Text>
          <Text style={styles.sectionContent}>하루 30분 걷기를 실천해보세요! 심혈관 건강에 좋습니다.</Text>
        </View>

        {/* 커뮤니티 게시판 섹션 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>커뮤니티 게시판</Text>
          <TouchableOpacity style={styles.button} onPress={() => alert('게시판으로 이동')}>
            <Text style={styles.buttonText}>게시판 보기</Text>
          </TouchableOpacity>
        </View>

        {/* 일일 운동 목표 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>일일 운동 목표</Text>
          <Text style={styles.sectionContent}>오늘 목표: 5000보 걷기</Text>
        </View>

        {/* 상담 요청 버튼 */}
        <View style={styles.section}>
          <TouchableOpacity style={styles.button} onPress={() => alert('상담 요청')}>
            <Text style={styles.buttonText}>헬스 상담 요청</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContainer: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 30,
    fontWeight: 'bold',
    color: '#2e3d49',
  },
  section: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
    color: '#333',
  },
  sectionContent: {
    fontSize: 16,
    color: '#666',
  },
  button: {
    backgroundColor: '#4CAF50',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 5,
    alignItems: 'center',
    marginTop: 15,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default IndexScreen;
