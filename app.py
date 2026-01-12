import React, { useState } from 'react';
import { BarChart3, Activity, TrendingUp, Users, MessageSquare, FileText, Calendar, Settings, LogOut, Menu, X, ChevronRight, Download, ClipboardList, BookOpen, Sparkles } from 'lucide-react';

const BrucellosisApp = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [activeTab, setActiveTab] = useState('login');
  const [language, setLanguage] = useState('English');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [predictionResult, setPredictionResult] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');

  // Mock data for dashboard stats
  const stats = {
    totalPredictions: 1247,
    positiveCases: 87,
    accuracyRate: 94.3,
    aiConsultations: 342
  };

  // Form state
  const [formData, setFormData] = useState({
    age: 5,
    breed: 'Select breed',
    sex: 'Select sex',
    calvings: 1,
    abortion: 'No',
    infertility: 'No',
    vaccination: 'Not Vaccinated',
    sample: 'Blood',
    test: 'RBPT',
    retained: 'No',
    disposal: 'Yes'
  });

  const translations = {
    English: {
      dashboard: 'Dashboard',
      newPrediction: 'New Prediction',
      history: 'History',
      analytics: 'Analytics',
      aiAssistant: 'AI Assistant',
      guidelines: 'Guidelines',
      settings: 'Settings',
      logout: 'Logout',
      welcome: 'Welcome to Brucellosis Prediction System',
      subtitle: 'AI-powered disease prediction and veterinary consultation'
    },
    Hindi: {
      dashboard: 'डैशबोर्ड',
      newPrediction: 'नई भविष्यवाणी',
      history: 'इतिहास',
      analytics: 'विश्लेषण',
      aiAssistant: 'AI सहायक',
      guidelines: 'दिशानिर्देश',
      settings: 'सेटिंग्स',
      logout: 'लॉगआउट',
      welcome: 'ब्रुसेलोसिस भविष्यवाणी प्रणाली में आपका स्वागत है',
      subtitle: 'AI-संचालित रोग भविष्यवाणी और पशु चिकित्सा परामर्श'
    }
  };

  const t = translations[language];

  const handlePrediction = () => {
    // Simulate prediction
    const mockResult = {
      status: Math.random() > 0.5 ? 'Positive' : 'Negative',
      confidence: (Math.random() * 30 + 70).toFixed(1),
      probabilities: {
        negative: Math.random() * 100,
        positive: Math.random() * 100,
        uncertain: Math.random() * 20
      }
    };
    setPredictionResult(mockResult);
  };

  const handleChatSend = () => {
    if (chatInput.trim()) {
      setChatHistory([...chatHistory, 
        { role: 'user', content: chatInput },
        { role: 'assistant', content: 'This is a simulated AI response. In the actual app, this would be powered by Gemini AI providing veterinary advice.' }
      ]);
      setChatInput('');
    }
  };

  // Login Screen
  if (!isLoggedIn) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-white rounded-2xl shadow-2xl overflow-hidden">
          <div className="bg-gradient-to-r from-emerald-500 to-teal-600 p-8 text-white">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-white rounded-xl flex items-center justify-center">
                <Activity className="text-emerald-600" size={28} />
              </div>
              <div>
                <h1 className="text-2xl font-bold">BrucellosisAI</h1>
                <p className="text-emerald-100 text-sm">Prediction System</p>
              </div>
            </div>
            <p className="text-emerald-50">{t.subtitle}</p>
          </div>

          <div className="p-8">
            <div className="flex gap-2 mb-6">
              <button
                onClick={() => setActiveTab('login')}
                className={`flex-1 py-2 rounded-lg font-medium transition-all ${
                  activeTab === 'login' 
                    ? 'bg-emerald-500 text-white' 
                    : 'bg-gray-100 text-gray-600'
                }`}
              >
                Login
              </button>
              <button
                onClick={() => setActiveTab('register')}
                className={`flex-1 py-2 rounded-lg font-medium transition-all ${
                  activeTab === 'register' 
                    ? 'bg-emerald-500 text-white' 
                    : 'bg-gray-100 text-gray-600'
                }`}
              >
                Register
              </button>
            </div>

            {activeTab === 'login' ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Email Address</label>
                  <input
                    type="email"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition"
                    placeholder="Enter your email"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                  <input
                    type="password"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition"
                    placeholder="Enter your password"
                  />
                </div>
                <button
                  onClick={() => setIsLoggedIn(true)}
                  className="w-full bg-gradient-to-r from-emerald-500 to-teal-600 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition-all"
                >
                  Login
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <input
                  type="text"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  placeholder="Full Name"
                />
                <input
                  type="email"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  placeholder="Email Address"
                />
                <input
                  type="tel"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  placeholder="Phone Number"
                />
                <input
                  type="text"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  placeholder="Location"
                />
                <input
                  type="password"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  placeholder="Password"
                />
                <button className="w-full bg-gradient-to-r from-emerald-500 to-teal-600 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition-all">
                  Register
                </button>
              </div>
            )}

            <div className="mt-6 flex justify-center gap-2 text-sm">
              <span className="text-gray-600">Language:</span>
              <button
                onClick={() => setLanguage('English')}
                className={`font-medium ${language === 'English' ? 'text-emerald-600' : 'text-gray-400'}`}
              >
                English
              </button>
              <span className="text-gray-400">/</span>
              <button
                onClick={() => setLanguage('Hindi')}
                className={`font-medium ${language === 'Hindi' ? 'text-emerald-600' : 'text-gray-400'}`}
              >
                हिंदी
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Main Dashboard
  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-20'} bg-white border-r border-gray-200 transition-all duration-300 flex flex-col`}>
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center">
              <Activity className="text-white" size={24} />
            </div>
            {sidebarOpen && (
              <div>
                <h1 className="text-lg font-bold text-gray-800">BrucellosisAI</h1>
                <p className="text-xs text-gray-500">Prediction System</p>
              </div>
            )}
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          {[
            { icon: BarChart3, label: t.dashboard },
            { icon: Activity, label: t.newPrediction },
            { icon: FileText, label: t.history },
            { icon: TrendingUp, label: t.analytics }
          ].map((item, idx) => (
            <button
              key={idx}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-emerald-50 text-gray-700 hover:text-emerald-600 transition-all group"
            >
              <item.icon size={20} />
              {sidebarOpen && <span className="font-medium">{item.label}</span>}
            </button>
          ))}

          <div className="pt-4 border-t border-gray-200 mt-4">
            <p className={`text-xs text-gray-400 px-4 mb-2 ${!sidebarOpen && 'hidden'}`}>RESOURCES</p>
            {[
              { icon: MessageSquare, label: t.aiAssistant, badge: 'AI' },
              { icon: BookOpen, label: t.guidelines },
              { icon: Settings, label: t.settings }
            ].map((item, idx) => (
              <button
                key={idx}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-50 text-gray-600 hover:text-gray-800 transition-all"
              >
                <item.icon size={20} />
                {sidebarOpen && (
                  <>
                    <span className="font-medium flex-1 text-left">{item.label}</span>
                    {item.badge && (
                      <span className="bg-emerald-100 text-emerald-700 text-xs px-2 py-0.5 rounded-full font-semibold">
                        {item.badge}
                      </span>
                    )}
                  </>
                )}
              </button>
            ))}
          </div>
        </nav>

        <div className="p-4 border-t border-gray-200">
          {sidebarOpen && (
            <div className="bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl p-4 mb-4">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="text-white" size={20} />
                <span className="text-white font-semibold text-sm">AI-Powered Insights</span>
              </div>
              <p className="text-emerald-50 text-xs mb-3">Get expert veterinary recommendations powered by advanced AI</p>
              <button className="w-full bg-white text-emerald-600 text-sm font-semibold py-2 rounded-lg hover:shadow-lg transition-all">
                Learn more →
              </button>
            </div>
          )}
          <button
            onClick={() => setIsLoggedIn(false)}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-red-50 text-gray-600 hover:text-red-600 transition-all"
          >
            <LogOut size={20} />
            {sidebarOpen && <span className="font-medium">{t.logout}</span>}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-8 py-4 flex items-center justify-between sticky top-0 z-10">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-lg transition"
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <div>
              <h2 className="text-xl font-bold text-gray-800">Brucellosis Prediction Dashboard</h2>
              <p className="text-sm text-gray-500">{t.subtitle}</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 outline-none"
            >
              <option>English</option>
              <option>हिंदी</option>
            </select>
            <div className="w-10 h-10 bg-emerald-100 rounded-full flex items-center justify-center text-emerald-600 font-semibold">
              U
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="p-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-all">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Total Predictions</p>
                  <h3 className="text-3xl font-bold text-gray-800">{stats.totalPredictions}</h3>
                </div>
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                  <BarChart3 className="text-blue-600" size={24} />
                </div>
              </div>
              <div className="flex items-center gap-1 text-sm">
                <TrendingUp size={16} className="text-green-600" />
                <span className="text-green-600 font-medium">+12.5%</span>
                <span className="text-gray-500">vs last month</span>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-all">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Positive Cases</p>
                  <h3 className="text-3xl font-bold text-gray-800">{stats.positiveCases}</h3>
                </div>
                <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center">
                  <Activity className="text-red-600" size={24} />
                </div>
              </div>
              <div className="flex items-center gap-1 text-sm">
                <TrendingUp size={16} className="text-red-600" />
                <span className="text-red-600 font-medium">+3.2%</span>
                <span className="text-gray-500">vs last month</span>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-all">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Accuracy Rate</p>
                  <h3 className="text-3xl font-bold text-gray-800">{stats.accuracyRate}%</h3>
                </div>
                <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                  <TrendingUp className="text-green-600" size={24} />
                </div>
              </div>
              <div className="flex items-center gap-1 text-sm">
                <TrendingUp size={16} className="text-green-600" />
                <span className="text-green-600 font-medium">+1.8%</span>
                <span className="text-gray-500">vs last month</span>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-all">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-sm text-gray-600 mb-1">AI Consultations</p>
                  <h3 className="text-3xl font-bold text-gray-800">{stats.aiConsultations}</h3>
                </div>
                <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
                  <Sparkles className="text-purple-600" size={24} />
                </div>
              </div>
              <div className="flex items-center gap-1 text-sm">
                <TrendingUp size={16} className="text-purple-600" />
                <span className="text-purple-600 font-medium">+24.1%</span>
                <span className="text-gray-500">vs last month</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Prediction Form */}
            <div className="lg:col-span-2 bg-white rounded-xl p-6 border border-gray-200">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-800">Animal Information Input</h3>
                <button className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition text-sm font-medium flex items-center gap-2">
                  <Download size={16} />
                  Save Template
                </button>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Age (Years)</label>
                  <input
                    type="number"
                    value={formData.age}
                    onChange={(e) => setFormData({...formData, age: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                    placeholder="Enter age"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Breed/Species</label>
                  <select
                    value={formData.breed}
                    onChange={(e) => setFormData({...formData, breed: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  >
                    <option>Select breed</option>
                    <option>Cattle</option>
                    <option>Buffalo</option>
                    <option>Goat</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Sex</label>
                  <select
                    value={formData.sex}
                    onChange={(e) => setFormData({...formData, sex: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  >
                    <option>Select sex</option>
                    <option>Male</option>
                    <option>Female</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Number of Calvings</label>
                  <input
                    type="number"
                    value={formData.calvings}
                    onChange={(e) => setFormData({...formData, calvings: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Abortion History</label>
                  <select
                    value={formData.abortion}
                    onChange={(e) => setFormData({...formData, abortion: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  >
                    <option>No</option>
                    <option>Yes</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Vaccination Status</label>
                  <select
                    value={formData.vaccination}
                    onChange={(e) => setFormData({...formData, vaccination: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  >
                    <option>Not Vaccinated</option>
                    <option>Vaccinated</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Sample Type</label>
                  <select
                    value={formData.sample}
                    onChange={(e) => setFormData({...formData, sample: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  >
                    <option>Blood</option>
                    <option>Serum</option>
                    <option>Milk</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Test Type</label>
                  <select
                    value={formData.test}
                    onChange={(e) => setFormData({...formData, test: e.target.value})}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  >
                    <option>RBPT</option>
                    <option>ELISA</option>
                    <option>MRT</option>
                  </select>
                </div>
              </div>

              <button
                onClick={handlePrediction}
                className="w-full bg-gradient-to-r from-emerald-500 to-teal-600 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition-all flex items-center justify-center gap-2"
              >
                <Activity size={20} />
                Run AI Prediction
              </button>

              {predictionResult && (
                <div className={`mt-6 p-6 rounded-xl border-l-4 ${
                  predictionResult.status === 'Positive' 
                    ? 'bg-red-50 border-red-500' 
                    : 'bg-green-50 border-green-500'
                }`}>
                  <h4 className="text-lg font-bold mb-2">
                    Prediction Results: <span className={predictionResult.status === 'Positive' ? 'text-red-600' : 'text-green-600'}>
                      {predictionResult.status}
                    </span>
                  </h4>
                  <p className="text-gray-700">
                    Confidence Score: <strong>{predictionResult.confidence}%</strong>
                  </p>
                  <div className="mt-4">
                    <p className="text-sm font-medium text-gray-700 mb-2">Probability Distribution</p>
                    <div className="space-y-2">
                      {Object.entries(predictionResult.probabilities).map(([key, value]) => (
                        <div key={key}>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="capitalize">{key}</span>
                            <span>{value.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                key === 'negative' ? 'bg-green-500' : key === 'positive' ? 'bg-red-500' : 'bg-yellow-500'
                              }`}
                              style={{ width: `${value}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Sidebar */}
            <div className="space-y-6">
              {/* AI Assistant Card */}
              <div className="bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl p-6 text-white">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 bg-white rounded-xl flex items-center justify-center">
                    <MessageSquare className="text-emerald-600" size={24} />
                  </div>
                  <div className="flex-1">
                    <h4 className="font-bold text-lg">Veterinary AI Assistant</h4>
                    <span className="text-emerald-100 text-sm flex items-center gap-1">
                      <Sparkles size={12} />
                      AI Powered
                    </span>
                  </div>
                </div>
                <p className="text-emerald-50 text-sm mb-4">
                  Get instant expert advice on brucellosis, animal health, and milk safety
                </p>
                <button
                  onClick={() => setShowChatbot(!showChatbot)}
                  className="w-full bg-white text-emerald-600 py-2 rounded-lg font-semibold hover:shadow-lg transition-all flex items-center justify-center gap-2"
                >
                  <MessageSquare size={18} />
                  Start Consultation
                </button>
              </div>

              {/* Quick Actions */}
              <div className="bg-white rounded-xl p-6 border border-gray-200">
                <h4 className="font-bold text-gray-800 mb-4">Quick Actions</h4>
                <div className="space-y-2">
                  {[
                    { icon: Download, label: 'Export Report' },
                    { icon: ClipboardList, label: 'Schedule Test' },
                    { icon: BookOpen, label: 'View Guidelines' }
                  ].map((action, idx) => (
                    <button
                      key={idx}
                      className="w-full flex items-center justify-between px-4 py-3 rounded-lg hover:bg-gray-50 text-gray-700 transition-all group"
                    >
                      <div className="flex items-center gap-3">
                        <action.icon size={18} className="text-gray-500 group-hover:text-emerald-600 transition" />
                        <span className="font-medium">{action.label}</span>
                      </div>
                      <ChevronRight size={18} className="text-gray-400 group-hover:text-emerald-600 transition" />
                    </button>
                  ))}
                </div>
              </div>

              {/* Probability Chart */}
              {predictionResult && (
                <div className="bg-white rounded-xl p-6 border border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="font-bold text-gray-800">Probability Distribution</h4>
                    <select className="px-3 py-1 border border-gray-300 rounded-lg text-sm outline-none">
                      <option>Last 7 days</option>
                      <option>Last 30 days</option>
                    </select>
                  </div>
                  <div className="space-y-4">
                    {Object.entries(predictionResult.probabilities).map(([key, value]) => (
                      <div key={key}>
                        <div className="flex justify-between text-sm mb-2">
                          <span className="capitalize font-medium text-gray-700">{key}</span>
                          <span className="font-semibold">{value.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                          <div
                            className={`h-3 rounded-full transition-all duration-500 ${
                              key === 'negative' ? 'bg-green-500' : 
                              key === 'positive' ? 'bg-red-500' : 'bg-yellow-500'
                            }`}
                            style={{ width: `${value}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Chatbot Modal */}
          {showChatbot && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
              <div className="bg-white rounded-2xl w-full max-w-2xl max-h-[600px] flex flex-col shadow-2xl">
                <div className="bg-gradient-to-r from-emerald-500 to-teal-600 p-6 rounded-t-2xl text-white flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-white rounded-xl flex items-center justify-center">
                      <MessageSquare className="text-emerald-600" size={24} />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold">Veterinary Assistant</h3>
                      <p className="text-emerald-100 text-sm">Ask questions about Brucellosis & animal health</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowChatbot(false)}
                    className="w-10 h-10 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30 transition flex items-center justify-center"
                  >
                    <X className="text-white" size={20} />
                  </button>
                </div>

                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                  {chatHistory.length === 0 ? (
                    <div className="text-center py-12">
                      <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <MessageSquare className="text-emerald-600" size={32} />
                      </div>
                      <h4 className="text-lg font-semibold text-gray-800 mb-2">Start a Conversation</h4>
                      <p className="text-gray-500 text-sm">Ask me anything about Brucellosis, animal health, or milk safety</p>
                    </div>
                  ) : (
                    chatHistory.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-[80%] rounded-2xl p-4 ${
                            msg.role === 'user'
                              ? 'bg-gradient-to-br from-emerald-500 to-teal-600 text-white'
                              : 'bg-gray-100 text-gray-800'
                          }`}
                        >
                          <p className="text-sm font-semibold mb-1">
                            {msg.role === 'user' ? 'You' : 'AI Assistant'}
                          </p>
                          <p className="text-sm">{msg.content}</p>
                        </div>
                      </div>
                    ))
                  )}
                </div>

                <div className="p-6 border-t border-gray-200">
                  <div className="flex gap-3">
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleChatSend()}
                      placeholder="Type your question here..."
                      className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                    />
                    <button
                      onClick={handleChatSend}
                      className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-lg font-semibold hover:shadow-lg transition-all"
                    >
                      Send
                    </button>
                  </div>
                  <button
                    onClick={() => setChatHistory([])}
                    className="mt-3 text-sm text-gray-500 hover:text-gray-700 transition"
                  >
                    Clear Chat
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-white border-t border-gray-200 px-8 py-4 text-center text-sm text-gray-500">
          Developed for Veterinary Health Solutions
        </div>
      </div>
    </div>
  );
};

export default BrucellosisApp;
